import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import snntorch as snn
from snntorch import surrogate
import transformers
transformers.logging.set_verbosity_error()

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpikeEncoder:
    def __init__(self, model_name='bert-base-uncased', num_neurons_per_dim=1000, max_rate=1000, embedding_file='ada002_embeddings.pt'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embeddings = torch.load(embedding_file)
        self.num_neurons_per_dim = num_neurons_per_dim
        self.max_rate = max_rate

    def encode(self, text, time_steps=100):
        # Tokenize input text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Get ada-002 embeddings for tokens
        embeddings = torch.stack([torch.tensor(self.embeddings[self.tokenizer.decode([token])]) for token in tokens])
        
        # Generate population code
        spike_trains = self.generate_population_code(embeddings, time_steps)
        return spike_trains

    def generate_population_code(self, embeddings, time_steps):
        seq_len, embedding_dim = embeddings.shape
        
        # Normalize embeddings to [0, 1]
        norm_embeddings = (F.normalize(embeddings, p=2, dim=-1) + 1) / 2
        
        # Create preferred values for each neuron in the population
        preferred_values = torch.linspace(0, 1, self.num_neurons_per_dim).to(device)
        preferred_values = preferred_values.unsqueeze(0).unsqueeze(0).expand(seq_len, embedding_dim, -1)
        
        # Calculate firing rates based on distance to preferred values
        distances = (norm_embeddings.unsqueeze(-1) - preferred_values).pow(2)
        firing_rates = self.max_rate * torch.exp(-distances / 0.1)
        
        # Incorporate temporal positional encoding
        position_offset = torch.arange(seq_len).unsqueeze(1).unsqueeze(2).float() / seq_len
        firing_rates = firing_rates * (1 - position_offset)
        
        spikes = torch.rand_like(firing_rates.unsqueeze(-1).expand(-1, -1, -1, time_steps)) < (firing_rates.unsqueeze(-1) / self.max_rate)
        return spikes.float()

class SpikeDecoder:
    def __init__(self, model_name='bert-base-uncased', num_neurons_per_dim=1000, embedding_file='ada002_embeddings.pt'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embeddings = torch.load(embedding_file)
        self.num_neurons_per_dim = num_neurons_per_dim

    def decode(self, spike_trains):
        # Decode population code
        decoded_embeddings = self.decode_population_code(spike_trains)
        
        # Convert embeddings to text
        reconstructed_text = self.embeddings_to_text(decoded_embeddings)
        
        return reconstructed_text

    def decode_population_code(self, spike_trains):
        # Sum spikes across time dimension
        spike_counts = spike_trains.sum(dim=-1)
        
        # Create preferred values
        preferred_values = torch.linspace(0, 1, self.num_neurons_per_dim).to(spike_trains.device)
        preferred_values = preferred_values.unsqueeze(0).unsqueeze(0)
        
        # Calculate weighted average of preferred values
        decoded_values = (spike_counts * preferred_values).sum(dim=-1) / (spike_counts.sum(dim=-1) + 1e-8)
        
        # Rescale decoded values to [-1, 1]
        decoded_values = decoded_values * 2 - 1
        
        # Normalize the decoded values
        decoded_values = F.normalize(decoded_values, p=2, dim=-1)
        
        return decoded_values

    def embeddings_to_text(self, embeddings):
        tokens = []
        for embedding in embeddings:
            similarities = F.cosine_similarity(embedding.unsqueeze(0), torch.stack(list(self.embeddings.values())))
            closest_token = list(self.embeddings.keys())[similarities.argmax()]
            tokens.append(closest_token)
        
        # Remove special tokens and join
        cleaned_tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
        reconstructed_text = ' '.join(cleaned_tokens)
        
        return reconstructed_text

class SpikingLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.lif = snn.Leaky(beta=beta)

    def forward(self, x):
        # x: [batch_size, in_features, time_steps]
        mem = self.lif.init_leaky()
        spk_rec = []
        for t in range(x.shape[-1]):
            syn = self.fc(x[..., t])
            spk, mem = self.lif(syn, mem)
            spk_rec.append(spk.unsqueeze(-1))
        return torch.cat(spk_rec, dim=-1)

class PhaseShiftedSpikingSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.q_layer = SpikingLinearLayer(embedding_dim, embedding_dim)
        self.k_layer = SpikingLinearLayer(embedding_dim, embedding_dim)
        self.v_layer = SpikingLinearLayer(embedding_dim, embedding_dim)
        self.out_layer = SpikingLinearLayer(embedding_dim, embedding_dim)

    def forward(self, x):
        # x: [seq_len, embedding_dim, time_steps]
        seq_len, _, time_steps = x.shape
        
        q = self.q_layer(x).view(seq_len, self.num_heads, self.head_dim, time_steps)
        k = self.k_layer(x).view(seq_len, self.num_heads, self.head_dim, time_steps)
        v = self.v_layer(x).view(seq_len, self.num_heads, self.head_dim, time_steps)
        
        # Apply phase shift to different heads
        for h in range(self.num_heads):
            shift = h * (time_steps // self.num_heads)
            q[:, h] = torch.roll(q[:, h], shifts=shift, dims=-1)
            k[:, h] = torch.roll(k[:, h], shifts=shift, dims=-1)
            v[:, h] = torch.roll(v[:, h], shifts=shift, dims=-1)
        
        attention_scores = self.coincidence_detection(q, k)
        
        # Normalize attention scores
        attention_weights = self.normalize_attention_scores(attention_scores)
        
        # Apply attention weights to values
        attended_values = self.apply_attention_weights(attention_weights, v)
        
        # Combine heads and pass through output layer
        attended_values = attended_values.view(seq_len, -1, time_steps)
        output = self.out_layer(attended_values)
        
        return output

    def coincidence_detection(self, q, k):
        # q, k: [seq_len, num_heads, head_dim, time_steps]
        seq_len, num_heads, head_dim, time_steps = q.shape
        attention_scores = torch.zeros(seq_len, num_heads, seq_len).to(q.device)
        
        for i in range(seq_len):
            q_i = q[i].unsqueeze(1)
            coincidence = (q_i * k).sum(dim=2)
            attention_scores[i] = coincidence.sum(dim=-1)
        
        return attention_scores

    def normalize_attention_scores(self, attention_scores):
        # attention_scores: [seq_len, num_heads, seq_len]
        return F.softmax(attention_scores, dim=-1)

    def apply_attention_weights(self, attention_weights, v):
        # attention_weights: [seq_len, num_heads, seq_len]
        # v: [seq_len, num_heads, head_dim, time_steps]
        return torch.einsum('snh,shdt->sndt', attention_weights, v)

class SpikingTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
        self.self_attention = PhaseShiftedSpikingSelfAttention(embedding_dim, num_heads)
        self.feedforward = nn.Sequential(
            SpikingLinearLayer(embedding_dim, embedding_dim * 4),
            SpikingLinearLayer(embedding_dim * 4, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Self-attention
        attended = self.self_attention(x)
        x = self.layer_norm1(x + attended)
        
        # Feedforward
        ff_output = self.feedforward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x

class SpikingTransformer(nn.Module):
    def __init__(self, embedding_dim, num_layers=6, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([SpikingTransformerLayer(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SpikingTransformerModel:
    def __init__(self, model_name='bert-base-uncased', num_neurons_per_dim=1000, embedding_dim=768, num_layers=6, num_heads=8, chunk_size=512, overlap=256):
        self.encoder = SpikeEncoder(model_name, num_neurons_per_dim)
        self.transformer = SpikingTransformer(embedding_dim, num_layers, num_heads).to(device)
        self.decoder = SpikeDecoder(model_name, num_neurons_per_dim)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def forward(self, text):
        # Encode text to spike trains
        spike_trains = self.encoder.encode(text)
        
        # Process in chunks with overlap
        seq_len = spike_trains.shape[0]
        chunks = []
        for i in range(0, seq_len, self.chunk_size - self.overlap):
            chunk = spike_trains[i:i+self.chunk_size]
            if chunk.shape[0] < self.chunk_size:
                # Pad the last chunk if necessary
                padding = torch.zeros(self.chunk_size - chunk.shape[0], chunk.shape[1], chunk.shape[2], chunk.shape[3]).to(device)
                chunk = torch.cat([chunk, padding], dim=0)
            transformed_chunk = self.transformer(chunk)
            chunks.append(transformed_chunk)
        
        # Combine chunks
        if len(chunks) > 1:
            combined = chunks[0][:-self.overlap]
            for i in range(1, len(chunks) - 1):
                combined = torch.cat([combined, chunks[i][self.overlap:-self.overlap]], dim=0)
            combined = torch.cat([combined, chunks[-1][self.overlap:seq_len-i*(self.chunk_size-self.overlap)]], dim=0)
        else:
            combined = chunks[0][:seq_len]
        
        reconstructed_text = self.decoder.decode(combined)
        return reconstructed_text

def test_spiking_transformer(model_name='bert-base-uncased', num_neurons_per_dim=1000):
    model = SpikingTransformerModel(model_name, num_neurons_per_dim)

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming technology and society.",
        "In the realm of quantum mechanics, particles behave in counterintuitive ways.",
        "This is a longer text to test the model's ability to handle larger inputs. " * 10  # Repeated 10 times for a longer input
    ]

    for text in test_texts:
        print(f"\nOriginal text: {text}")
        reconstructed_text = model.forward(text)
        print(f"Reconstructed text: {reconstructed_text}")

if __name__ == "__main__":
    test_spiking_transformer(model_name='bert-base-uncased', num_neurons_per_dim=2500)