import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from scipy.spatial.distance import cosine
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from typing import Dict, Union, List
import logging
import random
import struct
import gzip
import h5py
from transformers import AutoTokenizer, AutoModel

# Initialize OpenAI client (using environment variable)
client = OpenAI(
  api_key=OPENAI_API_KEY)

# Set up CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embeddings(texts: Union[str, List[str]], model: str = "text-embedding-ada-002") -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]
    try:
        response = client.embeddings.create(input=texts, model=model)
        logger.info(f"Successfully got embeddings for {len(texts)} texts")
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        return []

def generate_population_code(embeddings, num_neurons_per_dim=5000, max_rate=1000, time_steps=100):
    batch_size, embedding_dim = embeddings.shape
    
    # Normalize embeddings to [-1, 1]
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Create preferred values for each neuron in the population
    preferred_values = torch.linspace(-1, 1, num_neurons_per_dim).to(device).unsqueeze(0).unsqueeze(0)
    preferred_values = preferred_values.repeat(batch_size, embedding_dim, 1)
    
    # Reshape embeddings to match preferred_values shape
    embeddings_expanded = norm_embeddings.unsqueeze(-1).repeat(1, 1, num_neurons_per_dim)
    
    # Calculate firing rates based on distance to preferred values
    distances = (embeddings_expanded - preferred_values).pow(2)
    firing_rates = max_rate * torch.exp(-distances / 0.2)  # Reduced tuning curve width
    
    # Ensure the output is 3D: [embedding_dim, num_neurons_per_dim, time_steps]
    firing_rates = firing_rates.squeeze(0).unsqueeze(-1).repeat(1, 1, time_steps)
    
    return firing_rates, num_neurons_per_dim

def decode_population_code(firing_rates, num_neurons_per_dim=10):
    # If input is 3D, take the mean across the time dimension
    if firing_rates.dim() == 3:
        firing_rates = firing_rates.mean(dim=-1)
    
    # Create preferred values
    preferred_values = torch.linspace(-1, 1, num_neurons_per_dim).to(device)
    
    # Calculate weighted average of preferred values
    decoded_values = (firing_rates * preferred_values).sum(dim=-1) / (firing_rates.sum(dim=-1) + 1e-8)
    
    # Normalize the decoded values
    decoded_values = F.normalize(decoded_values, p=2, dim=0)
    
    return decoded_values

def visualize_spike_train(word, spike_train, save_dir='spike_train_plots', max_dims=10):
    if isinstance(spike_train, torch.Tensor):
        spike_train = spike_train.cpu().numpy()
    
    if spike_train.ndim == 4:
        spike_train = spike_train.squeeze(0)  # Remove the batch dimension if present
    
    embedding_dim, num_neurons_per_dim, time_steps = spike_train.shape
    
    # Limit the number of dimensions to visualize
    embedding_dim = min(embedding_dim, max_dims)
    
    fig, axes = plt.subplots(embedding_dim, 1, figsize=(20, 4 * embedding_dim))
    if embedding_dim == 1:
        axes = [axes]
    
    for i in range(embedding_dim):
        axes[i].imshow(spike_train[i], cmap='binary', aspect='auto')
        axes[i].set_title(f'Dimension {i+1}')
        axes[i].set_ylabel('Neuron')
        axes[i].set_xlabel('Time step')
    
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'{word}_spike_train.png')
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Spike train visualization for '{word}' saved in {save_path}")

def print_spike_train_stats(word, spike_train):
    if isinstance(spike_train, torch.Tensor):
        spike_train = spike_train.cpu().numpy()
    
    if spike_train.ndim == 4:
        spike_train = spike_train.squeeze(0)
    
    embedding_dim, num_neurons_per_dim, time_steps = spike_train.shape
    
    total_spikes = np.sum(spike_train)
    avg_firing_rate = total_spikes / (embedding_dim * num_neurons_per_dim * time_steps)

def process_batch(batch, embeddings_dict, time_steps, max_rate, num_neurons_per_dim):
    result = {}
    for word in batch:
        if word not in embeddings_dict:
            embedding = get_embeddings(word)[0]
            embeddings_dict[word] = embedding
        else:
            embedding = embeddings_dict[word]
        embedding_tensor = torch.tensor(embedding, device=device).unsqueeze(0)
        spike_train = generate_population_code(embedding_tensor, num_neurons_per_dim, max_rate, time_steps)
        result[word] = spike_train.squeeze(0)
    return result

def load_words_from_file(file_path):
    with open(file_path, 'r') as file:
        return [word.strip() for word in file.readlines()]

def load_embeddings_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def save_embeddings_to_pickle(embeddings_dict, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(embeddings_dict, file)

def encode_words_sequential(words, embeddings_dict, time_steps=100, max_rate=1000, num_neurons_per_dim=10):
    word_spike_dict = {}
    for word in tqdm(words, desc="Encoding words"):
        try:
            if word not in embeddings_dict:
                embedding = get_embeddings(word)[0]
                embeddings_dict[word] = embedding
            else:
                embedding = embeddings_dict[word]
            embedding_tensor = torch.tensor(embedding, device=device).unsqueeze(0)
            spike_train, neurons_per_dim = generate_population_code(embedding_tensor, num_neurons_per_dim, max_rate, time_steps)
            word_spike_dict[word] = (spike_train, neurons_per_dim)  # Store both spike train and neurons_per_dim
        except Exception as e:
            logger.error(f"Error encoding word '{word}': {str(e)}")
    
    logger.info(f"Encoded {len(word_spike_dict)} words out of {len(words)} total words.")
    if len(word_spike_dict) == 0:
        logger.warning("No words were successfully encoded.")
    elif len(word_spike_dict) < len(words):
        logger.warning(f"{len(words) - len(word_spike_dict)} words failed to encode.")
    
    return word_spike_dict

def spike_train_analogy(a, b, c, word_spike_dict, top_n=5):
    a_rates, a_neurons = word_spike_dict[a]
    b_rates, b_neurons = word_spike_dict[b]
    c_rates, c_neurons = word_spike_dict[c]
    
    # Decode firing rates
    a_emb = decode_population_code(a_rates, a_neurons)
    b_emb = decode_population_code(b_rates, b_neurons)
    c_emb = decode_population_code(c_rates, c_neurons)
    
    # Perform analogy operation in the embedding space
    result = a_emb - b_emb + c_emb
    result = F.normalize(result, p=2, dim=0)
    
    # Compute similarities using cosine similarity
    similarities = []
    for word, (firing_rates, num_neurons) in word_spike_dict.items():
        word_embedding = decode_population_code(firing_rates, num_neurons)
        sim = F.cosine_similarity(result.unsqueeze(0), word_embedding.unsqueeze(0))
        similarities.append((word, sim.item()))
    
    # Sort by similarity (descending) and get top N
    top_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    return top_results

def run_analogy_benchmark(analogies, word_spike_dict):
    correct = 0
    total = 0
    
    for a, b, c, expected in tqdm(analogies, desc="Running spike train analogies"):
        if a not in word_spike_dict or b not in word_spike_dict or c not in word_spike_dict or expected not in word_spike_dict:
            continue
        
        results = spike_train_analogy(a, b, c, word_spike_dict)
        
        if results and results[0][0] == expected:
            correct += 1
        total += 1
        
        print(f"\nAnalogy: {a} - {b} + {c} = {expected}")
        print(f"Top 5 results: {results}")
        
        # Print the rank of the expected answer in the results
        expected_rank = next((i for i, (word, _) in enumerate(results) if word == expected), None)
        if expected_rank is not None:
            print(f"Rank of expected answer '{expected}': {expected_rank + 1}")
        else:
            print(f"Expected answer '{expected}' not found in top {len(results)} results")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall spike train analogy task accuracy: {accuracy:.2%}")
    return accuracy

def continuous_embedding_analogy(a, b, c, embedding_dict, top_n=5):
    a_emb = torch.tensor(embedding_dict[a])
    b_emb = torch.tensor(embedding_dict[b])
    c_emb = torch.tensor(embedding_dict[c])
    
    result = a_emb - b_emb + c_emb
    
    similarities = []
    for word, emb in embedding_dict.items():
        sim = torch.cosine_similarity(result, torch.tensor(emb), dim=0)
        similarities.append((word, sim.item()))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def run_continuous_analogy_benchmark(analogies, embedding_dict):
    correct = 0
    total = 0
    
    for a, b, c, expected in tqdm(analogies, desc="Running continuous embedding analogies"):
        if a not in embedding_dict or b not in embedding_dict or c not in embedding_dict or expected not in embedding_dict:
            continue
        
        results = continuous_embedding_analogy(a, b, c, embedding_dict)
        
        if results and results[0][0] == expected:
            correct += 1
        total += 1
        
        print(f"\nAnalogy: {a} - {b} + {c} = {expected}")
        print(f"Top 5 results: {results}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall continuous embedding analogy task accuracy: {accuracy:.2%}")
    return accuracy

def load_or_compute_embeddings(binary_embeddings_file, embeddings_file, words_to_encode):
    word_spike_dict = None
    embeddings_dict = None

    if os.path.exists(binary_embeddings_file):
        print("Found existing binary embeddings file. Loading...")
        word_spike_dict = load_binary_embeddings(binary_embeddings_file, words_to_encode)
        if word_spike_dict is not None:
            print("Successfully loaded valid binary embeddings.")
            if os.path.exists(embeddings_file):
                embeddings_dict = load_embeddings_from_pickle(embeddings_file)
                if embeddings_dict is not None:
                    print("Successfully loaded embeddings from pickle.")
                    return word_spike_dict, embeddings_dict
                else:
                    print("Failed to load embeddings from pickle.")
            else:
                print(f"Embeddings file {embeddings_file} not found.")
        else:
            print("Failed to load binary embeddings.")
    else:
        print("Binary embeddings file not found.")

    print("Will compute new embeddings.")

    if os.path.exists(embeddings_file):
        print("Found existing embeddings file. Loading...")
        embeddings_dict = load_embeddings_from_pickle(embeddings_file)
        if embeddings_dict is not None:
            print(f"Loaded {len(embeddings_dict)} pre-computed embeddings.")
        else:
            print("Failed to load pre-computed embeddings.")
            embeddings_dict = {}
    else:
        print("No pre-computed embeddings found. Will use API calls.")
        embeddings_dict = {}

    print(f"Encoding {len(words_to_encode)} words...")
    word_spike_dict = encode_words_sequential(words_to_encode, embeddings_dict)
    print("Encoding complete.")

    if word_spike_dict and len(word_spike_dict) > 0:
        print("Saving binary embeddings...")
        save_binary_embeddings(word_spike_dict, binary_embeddings_file)
        print("Binary embeddings saved.")

        if not os.path.exists(embeddings_file) and embeddings_dict:
            print("Saving updated embeddings...")
            save_embeddings_to_pickle(embeddings_dict, embeddings_file)
            print("Embeddings saved.")
    else:
        print("No words were successfully encoded. Cannot proceed.")
        return None, None

    print(f"Successfully computed {len(word_spike_dict)} binary embeddings.")
    return word_spike_dict, embeddings_dict

def test_word_similarity(word_spike_dict, word1, word2):
    if word1 in word_spike_dict and word2 in word_spike_dict:
        embedding1 = decode_population_code(word_spike_dict[word1][0], word_spike_dict[word1][1])
        embedding2 = decode_population_code(word_spike_dict[word2][0], word_spike_dict[word2][1])
        similarity = torch.cosine_similarity(embedding1, embedding2, dim=0)
        print(f"Similarity between '{word1}' and '{word2}': {similarity.item():.4f}")
    else:
        print(f"Cannot compute similarity: '{word1}' or '{word2}' not found in word_spike_dict")

def load_binary_embeddings(file_path, expected_words):
    loaded_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        with gzip.open(file_path, 'rb') as f:
            total_words = struct.unpack('>I', f.read(4))[0]
            
            for _ in tqdm(range(total_words), desc="Loading binary embeddings"):
                word_len = struct.unpack('>I', f.read(4))[0]
                word = f.read(word_len).decode('utf-8')
                
                spike_train_shape = struct.unpack('>IIII', f.read(16))
                spike_train = np.frombuffer(f.read(np.prod(spike_train_shape[:3]) * 4), dtype=np.float32).reshape(spike_train_shape[:3])
                num_neurons = spike_train_shape[3]
                
                loaded_dict[word] = (torch.from_numpy(spike_train).float().to(device), num_neurons)

        print(f"\nLoaded {len(loaded_dict)} binary embeddings from {file_path}")

        # Validate loaded embeddings
        missing_words = [word for word in expected_words if word not in loaded_dict]
        if missing_words:
            print(f"Warning: The following words are missing: {missing_words}")
        
        return loaded_dict
    except Exception as e:
        print(f"Error loading binary embeddings: {str(e)}")
        return None

def save_binary_embeddings(word_spike_dict, file_path):
    total_words = len(word_spike_dict)
    start_time = time.time()
    
    with gzip.open(file_path, 'wb') as f:
        f.write(struct.pack('>I', total_words))
        
        for word, (spike_train, num_neurons) in tqdm(word_spike_dict.items(), total=total_words, desc="Saving binary embeddings"):
            word_bytes = word.encode('utf-8')
            f.write(struct.pack('>I', len(word_bytes)))
            f.write(word_bytes)
            
            spike_train_np = spike_train.cpu().numpy()
            f.write(struct.pack('>IIII', *spike_train_np.shape, num_neurons))
            f.write(spike_train_np.tobytes())

    total_time = time.time() - start_time
    print(f"\nSaved {total_words} binary embeddings to {file_path}")
    print(f"Total saving time: {total_time/60:.2f} minutes")

def analyze_population_coding(word_spike_dict):
    for word, (spike_train, num_neurons) in word_spike_dict.items():
        original = decode_population_code(spike_train, num_neurons)
        re_encoded, _ = generate_population_code(original.unsqueeze(0), num_neurons_per_dim=num_neurons)
        re_decoded = decode_population_code(re_encoded, num_neurons)
        similarity = F.cosine_similarity(original, re_decoded, dim=0)
        print(f"Word: {word}, Encoding-Decoding Similarity: {similarity.item():.4f}")

def compare_analogy_performance(analogies, word_spike_dict, embedding_dict):
    spike_correct = 0
    continuous_correct = 0
    total = 0
    
    for a, b, c, expected in analogies:
        if (a not in word_spike_dict or b not in word_spike_dict or 
            c not in word_spike_dict or expected not in word_spike_dict or
            a not in embedding_dict or b not in embedding_dict or 
            c not in embedding_dict or expected not in embedding_dict):
            print(f"Skipping analogy {a} - {b} + {c} = {expected} due to missing words")
            continue
        
        spike_results = spike_train_analogy(a, b, c, word_spike_dict)
        continuous_results = continuous_embedding_analogy(a, b, c, embedding_dict)
        
        if spike_results and spike_results[0][0] == expected:
            spike_correct += 1
        if continuous_results and continuous_results[0][0] == expected:
            continuous_correct += 1
        total += 1
        
        print(f"\nAnalogy: {a} - {b} + {c} = {expected}")
        print(f"Spike train top 5 results: {spike_results}")
        print(f"Continuous embedding top 5 results: {continuous_results}")
        
        # Print the rank of the expected answer in both results
        spike_rank = next((i for i, (word, _) in enumerate(spike_results) if word == expected), None)
        continuous_rank = next((i for i, (word, _) in enumerate(continuous_results) if word == expected), None)
        
        print(f"Rank of expected answer '{expected}':")
        print(f"  Spike train: {spike_rank + 1 if spike_rank is not None else 'Not in top 5'}")
        print(f"  Continuous: {continuous_rank + 1 if continuous_rank is not None else 'Not in top 5'}")
    
    spike_accuracy = spike_correct / total if total > 0 else 0
    continuous_accuracy = continuous_correct / total if total > 0 else 0
    
    print(f"\nOverall analogy task accuracy:")
    print(f"Spike train: {spike_accuracy:.2%}")
    print(f"Continuous embedding: {continuous_accuracy:.2%}")
    
    return spike_accuracy, continuous_accuracy

class SpikeDecoder:
    def __init__(self, num_neurons_per_dim=1000, embedding_dim=768, model_name='bert-base-uncased'):
        self.num_neurons_per_dim = num_neurons_per_dim
        self.embedding_dim = embedding_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def decode_population_code(self, spike_trains):
        # Sum spikes across time dimension
        spike_counts = spike_trains.sum(dim=-1)  # Shape: [batch_size, seq_len, embedding_dim, num_neurons_per_dim]

        # Create preferred values
        preferred_values = torch.linspace(0, 1, self.num_neurons_per_dim).to(spike_trains.device)  # Shape: [num_neurons_per_dim]
        preferred_values = preferred_values.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 1, num_neurons_per_dim]

        # Calculate weighted average of preferred values
        decoded_values = (spike_counts * preferred_values).sum(dim=-1) / (spike_counts.sum(dim=-1) + 1e-8)

        # decoded_values shape is now [batch_size, seq_len, embedding_dim]
        return decoded_values

    def embeddings_to_words(self, embeddings):
        # Find the closest words for the entire sequence
        with torch.no_grad():
            vocab_embeddings = self.model.embeddings.word_embeddings.weight
            embeddings = embeddings.to(vocab_embeddings.device)
            
            # Compute cosine similarity
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
            vocab_embeddings_norm = F.normalize(vocab_embeddings, p=2, dim=-1)
            similarities = torch.matmul(embeddings_norm.view(-1, self.embedding_dim), vocab_embeddings_norm.T)
            
            closest_word_ids = similarities.argmax(dim=-1)
            tokens = self.tokenizer.convert_ids_to_tokens(closest_word_ids)
            
            # Remove special tokens and join
            cleaned_tokens = [token for token in tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
            reconstructed_text = self.tokenizer.convert_tokens_to_string(cleaned_tokens)

        return reconstructed_text

# Main execution
if __name__ == "__main__":
    file_path = 'common_words.txt'
    words_to_encode = load_words_from_file(file_path)
    embeddings_file = 'embeddings.pkl'
    binary_embeddings_file = 'binary_embeddings.gz'
    force_recompute = True  # Set this to True to force recomputation of embeddings

    word_spike_dict, embedding_dict = load_or_compute_embeddings(binary_embeddings_file, embeddings_file, words_to_encode)

    if word_spike_dict is not None and embedding_dict is not None:
        # Define analogy tasks
        analogies = [
            ("king", "man", "woman", "queen"),
            ("france", "paris", "japan", "tokyo"),
            ("big", "bigger", "small", "smaller"),
            ("good", "best", "bad", "worst"),
            ("cat", "kitten", "dog", "puppy"),
        ]

        # Compare analogy performance
        spike_accuracy, continuous_accuracy = compare_analogy_performance(analogies, word_spike_dict, embedding_dict)

        print(f"\nSpike train analogy accuracy: {spike_accuracy:.2%}")
        print(f"Continuous embedding analogy accuracy: {continuous_accuracy:.2%}")

        # Test word similarity
        print("\nTesting word similarities:")
        test_word_similarity(word_spike_dict, "man", "woman")
        test_word_similarity(word_spike_dict, "king", "queen")
        test_word_similarity(word_spike_dict, "man", "king")
        test_word_similarity(word_spike_dict, "cat", "dog")
        test_word_similarity(word_spike_dict, "happy", "sad")

        # Analyze population coding quality
        print("\nAnalyzing population coding quality:")
        analyze_population_coding(word_spike_dict)
    else:
        print("Failed to load or compute embeddings. Exiting.")