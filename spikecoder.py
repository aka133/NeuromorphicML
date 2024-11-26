import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import snntorch as snn
from snntorch import surrogate
import torch
import json
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpikeEncoder:
    def __init__(self, model_name='bert-base-uncased', num_neurons_per_dim=1000, max_rate=1000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.num_neurons_per_dim = num_neurons_per_dim
        self.max_rate = max_rate

    def encode(self, text, time_steps=100):
        # Tokenize and get input embeddings
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            input_embeddings = self.model.embeddings(input_ids=inputs['input_ids'])
        embeddings = input_embeddings.squeeze(0)  # Remove batch dimension

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
        
        # Generate spikes
        spikes = torch.rand_like(firing_rates.unsqueeze(-1).expand(-1, -1, -1, time_steps)) < (firing_rates.unsqueeze(-1) / self.max_rate)
        
        return spikes.float()

class SpikeDecoder:
    def __init__(self, model_name='bert-base-uncased', num_neurons_per_dim=1000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
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
        with torch.no_grad():
            vocab_embeddings = self.model.embeddings.word_embeddings.weight
            embeddings = embeddings.to(vocab_embeddings.device)
            
            # Compute cosine similarity
            similarities = F.cosine_similarity(embeddings.unsqueeze(1), vocab_embeddings.unsqueeze(0), dim=-1)
            
            closest_word_ids = similarities.argmax(dim=-1)
            tokens = self.tokenizer.convert_ids_to_tokens(closest_word_ids)
            
            # Remove special tokens and join
            cleaned_tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
            reconstructed_text = self.tokenizer.convert_tokens_to_string(cleaned_tokens)
            
            # Clean up tokenization spaces manually
            reconstructed_text = reconstructed_text.replace(" ##", "").replace("##", "")
        
        return reconstructed_text

# Test the encoder and decoder
def test_spike_coding(model_name='bert-base-uncased', num_neurons_per_dim=1000):
    encoder = SpikeEncoder(model_name=model_name, num_neurons_per_dim=num_neurons_per_dim)
    decoder = SpikeDecoder(model_name=model_name, num_neurons_per_dim=num_neurons_per_dim)

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming technology and society.",
        "In the realm of quantum mechanics, particles behave in counterintuitive ways."
    ]

    for text in test_texts:
        print(f"\nOriginal text: {text}")
        spike_trains = encoder.encode(text)
        reconstructed_text = decoder.decode(spike_trains)
        print(f"Reconstructed text: {reconstructed_text}")

        # Print some statistics about the spike trains
        print(f"Spike train shape: {spike_trains.shape}")
        print(f"Average firing rate: {spike_trains.mean().item():.4f}")
        print(f"Sparsity: {(spike_trains == 0).float().mean().item():.4f}")

if __name__ == "__main__":
    # You can change the model and number of neurons per dimension here
    test_spike_coding(model_name='bert-base-uncased', num_neurons_per_dim=2500)