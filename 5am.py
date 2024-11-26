import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Importing necessary libraries
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, DistributedSampler
import matplotlib.pyplot as plt
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import traceback

# Importing Hugging Face libraries
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict

# Benchmark Datasets
import pandas as pd
from scipy.stats import spearmanr
import requests
import zipfile


# Download and extract the WordSim-353 dataset
def download_simlex999_dataset():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    dataset_url = 'https://fh295.github.io/SimLex-999.zip'
    zip_path = 'SimLex-999.zip'
    dataset_dir = 'SimLex-999'
    dataset_file = os.path.join(dataset_dir, 'SimLex-999.txt')
    
    if not os.path.exists(dataset_file):
        print("Downloading SimLex-999 dataset...")
        response = requests.get(dataset_url)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("Extraction complete.")
    else:
        print("SimLex-999 dataset already exists.")
    
    return dataset_file

def load_simlex999_dataset():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import pandas as pd
    
    dataset_file = download_simlex999_dataset()
    
    # Read the dataset
    similarity_df = pd.read_csv(dataset_file, sep='\t')
    
    # Rename columns for consistency
    similarity_df = similarity_df.rename(columns={
        'word1': 'Word 1',
        'word2': 'Word 2',
        'SimLex999': 'Human (mean)'
    })
    
    return similarity_df


# Download and load the Google Analogy Test Set
def download_google_analogy_dataset():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    dataset_url = 'https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt'
    dataset_path = 'questions-words.txt'

    if not os.path.exists(dataset_path):
        print("Downloading Google Analogy dataset...")
        response = requests.get(dataset_url)
        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        print("Google Analogy dataset downloaded.")
    else:
        print("Google Analogy dataset already exists.")

    analogy_questions = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if not line.startswith(':'):
                words = line.strip().lower().split()
                if len(words) == 4:
                    analogy_questions.append(words)
    return analogy_questions

def get_semantic_loss_weight(epoch, total_epochs, initial_weight=15.0, final_weight=50.0):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return initial_weight * (final_weight / initial_weight) ** (epoch / total_epochs)

# Set seeds for reproducibility
def set_seed(seed):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DDP setup and cleanup functions
def setup(rank, world_size):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import torch.distributed as dist  # Import inside the function
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' if 'nccl' is not supported
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import torch.distributed as dist  # Import inside the function
    dist.destroy_process_group()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
vocab_size = len(tokenizer)

# Model weight initialization
def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.01)

# Dataset preparation function
def prepare_datasets():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Define the paths where the tokenized datasets will be saved
    tokenized_dataset_path = 'tokenized_dataset'
    train_dataset_path = 'tokenized_dataset_train'
    val_dataset_path = 'tokenized_dataset_val'

    # Check if the tokenized datasets exist
    if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path):
        print("Tokenized train and validation datasets already exist.")
        return
    else:
        if os.path.exists(tokenized_dataset_path):
            print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
            tokenized_dataset = load_from_disk(tokenized_dataset_path)
            print("Tokenized dataset loaded.")
        else:
            print("Tokenized dataset not found.")
            print("Loading OpenWebText dataset...")
            dataset = load_dataset('openwebtext', trust_remote_code=True)

            # Define a tokenization function
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=16,
                )

            # Apply the tokenization to the dataset
            print("Tokenizing the dataset...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['text']
            )

            # Save the tokenized dataset
            print(f"Saving tokenized dataset to {tokenized_dataset_path}...")
            tokenized_dataset.save_to_disk(tokenized_dataset_path)
            print("Tokenized dataset saved.")

        # Set the format for PyTorch
        tokenized_dataset.set_format(type='torch', columns=['input_ids'])

        # Split the dataset into training and validation sets using train_test_split()
        print("Splitting the dataset into training and validation sets...")
        split_dataset = tokenized_dataset['train'].train_test_split(
            test_size=0.1,  # 10% for validation
            seed=42,
            shuffle=True
        )
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']

        # Save the train and validation datasets separately
        print(f"Saving train dataset to {train_dataset_path}...")
        train_dataset.save_to_disk(train_dataset_path)
        print(f"Saving validation dataset to {val_dataset_path}...")
        val_dataset.save_to_disk(val_dataset_path)
        print("Train and validation datasets saved.")

# Model Components
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=16):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask=None):
        return self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

class BinaryProjection(nn.Module):
    def __init__(self, embed_dim, bit_lengths):
        super(BinaryProjection, self).__init__()
        self.bit_lengths = bit_lengths  # List of bit lengths, e.g., [1024, 2048, 4096, 8192]
        self.projections = nn.ModuleList()
        for bit_length in bit_lengths:
            fc1 = nn.Linear(embed_dim, bit_length)
            self.projections.append(fc1)

    def forward(self, x):
        binarized_embeddings = []
        for projection in self.projections:
            z = projection(x)
            z = torch.tanh(z)
            z_binary = BinarizationSTE.apply(z)
            binarized_embeddings.append(z_binary)
        return binarized_embeddings

class BinarizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        result = input.sign()
        print(f"Binarization output: min={result.min().item():.2f}, max={result.max().item():.2f}, mean={result.mean().item():.2f}")
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BinaryEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, bit_lengths, max_seq_length=16, dropout=0.1):
        super(BinaryEmbeddingModel, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, hidden_dim, num_layers, dropout)
        self.binarizer = BinaryProjection(embed_dim, bit_lengths)

        # Store bit_lengths as an instance attribute
        self.bit_lengths = bit_lengths

        # Create decoders for each bit length
        self.decoders = nn.ModuleList([
            Decoder(bit_length, embed_dim) for bit_length in bit_lengths
        ])

    def forward(self, x, src_key_padding_mask=None):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        binarized_embeddings = self.binarizer(x)

        reconstructed_embeddings = []
        for z_binary, decoder in zip(binarized_embeddings, self.decoders):
            z_reconstructed = decoder(z_binary)
            reconstructed_embeddings.append(z_reconstructed)

        # Both during training and validation, return all necessary outputs
        return binarized_embeddings, reconstructed_embeddings, x, x  # Include 'x' as labels

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, bit_length, embed_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(bit_length, bit_length * 2),
            nn.ReLU(),
            nn.Linear(bit_length * 2, embed_dim)
        )

    def forward(self, x):
        return self.fc(x)

def distance_preservation_loss(original_embeddings, binarized_embeddings, num_pairs=32):
    # Flatten embeddings
    orig_embeddings_flat = original_embeddings.view(-1, original_embeddings.size(-1))  # [N, embed_dim]
    bin_embeddings_flat = binarized_embeddings.view(-1, binarized_embeddings.size(-1))  # [N, embed_dim]
    N = orig_embeddings_flat.size(0)
    device = orig_embeddings_flat.device

    # Initialize lists to store distances
    cont_distances = []
    bin_distances = []

    # Loop over pairs
    for _ in range(num_pairs):
        idx1 = torch.randint(0, N, (1,), device=device)
        idx2 = torch.randint(0, N, (1,), device=device)

        # Get the embeddings
        orig_emb_1 = orig_embeddings_flat[idx1]
        orig_emb_2 = orig_embeddings_flat[idx2]
        bin_emb_1 = bin_embeddings_flat[idx1]
        bin_emb_2 = bin_embeddings_flat[idx2]

        # Compute distances
        cont_distance = torch.norm(orig_emb_1 - orig_emb_2, p=2)
        bin_distance = torch.norm(bin_emb_1 - bin_emb_2, p=2)

        # Append to lists
        cont_distances.append(cont_distance)
        bin_distances.append(bin_distance)

    # Stack distances
    cont_distances = torch.stack(cont_distances)  # [num_pairs]
    bin_distances = torch.stack(bin_distances)    # [num_pairs]

    # Compute loss
    loss = torch.nn.functional.mse_loss(bin_distances, cont_distances)
    return loss

def diversity_loss(embeddings, num_pairs=500):
    device = embeddings.device
    N = embeddings.size(0)

    # Ensure indices are of type torch.long and on the correct device
    idx1 = torch.randint(0, N, (num_pairs,), device=device, dtype=torch.long)
    offset = torch.randint(1, N, (num_pairs,), device=device, dtype=torch.long)
    idx2 = (idx1 + offset) % N

    # Gather pairs of embeddings
    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]

    # Compute diversity loss (e.g., negative cosine similarity)
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
    loss = torch.mean(cos_sim)

    return loss

def contrastive_loss(embeddings, labels, margin=1.0, num_pairs=32):
    N, seq_len, embed_dim = embeddings.size()
    if N < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Flatten embeddings and labels
    embeddings = embeddings.view(-1, embed_dim)  # [N*seq_len, embed_dim]
    labels = labels.view(-1)  # [N*seq_len]
    N = embeddings.size(0)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Sample pairs
    anchor_indices = torch.randint(0, N, (num_pairs,), device=embeddings.device)
    positive_indices = torch.randint(0, N, (num_pairs,), device=embeddings.device)
    negative_indices = torch.randint(0, N, (num_pairs,), device=embeddings.device)

    # Ensure positive pairs have the same label
    positive_indices = torch.where(labels[anchor_indices] == labels[positive_indices], 
                                   positive_indices, anchor_indices)

    # Ensure negative pairs have different labels
    negative_indices = torch.where(labels[anchor_indices] != labels[negative_indices], 
                                   negative_indices, (negative_indices + 1) % N)

    # Compute distances
    anchor_embeddings = embeddings[anchor_indices]
    positive_embeddings = embeddings[positive_indices]
    negative_embeddings = embeddings[negative_indices]

    positive_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
    negative_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)

    losses = F.relu(positive_distances - negative_distances + margin)
    return losses.mean()

def semantic_similarity_loss(embeddings, context_window=5):
    batch_size, seq_length, embed_dim = embeddings.size()
    device = embeddings.device
    loss_fn = nn.CosineEmbeddingLoss(reduction='mean')

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=2, eps=1e-8)  # [batch_size, seq_length, embed_dim]

    # Prepare indices
    seq_indices = torch.arange(seq_length, device=device, dtype=torch.long)

    total_loss = 0
    count = 0

    for i in range(seq_length):
        anchor = embeddings[:, i, :]  # [batch_size, embed_dim]

        # Convert 'i' to tensor on the same device and dtype
        i_tensor = torch.tensor(i, device=device, dtype=seq_indices.dtype)

        # Positive indices within context window, excluding anchor
        pos_indices = seq_indices[max(0, i - context_window): min(seq_length, i + context_window + 1)]
        pos_indices = pos_indices[pos_indices != i_tensor]

        if pos_indices.numel() == 0:
            # No positive indices available, skip
            continue

        # Randomly select a positive index
        pos_idx = pos_indices[torch.randint(0, pos_indices.numel(), (1,), device=device)]
        pos_idx = pos_idx.item()
        positive = embeddings[:, pos_idx, :]  # [batch_size, embed_dim]

        # Positive loss
        target = torch.ones(batch_size, device=device)
        positive_loss = loss_fn(anchor, positive, target)
        total_loss += positive_loss
        count += 1

        # Negative indices outside context window
        neg_indices = torch.cat((
            seq_indices[:max(0, i - context_window)],
            seq_indices[min(seq_length, i + context_window + 1):]
        ))

        if neg_indices.numel() == 0:
            # No negative indices available, skip
            continue

        # Convert 'i' to tensor on the same device and dtype for negatives
        # (Though not needed here as 'i' is not used in comparison with 'neg_indices')

        # Randomly select a negative index
        neg_idx = neg_indices[torch.randint(0, neg_indices.numel(), (1,), device=device)]
        neg_idx = neg_idx.item()
        negative = embeddings[:, neg_idx, :]  # [batch_size, embed_dim]

        # Negative loss
        target = -torch.ones(batch_size, device=device)
        negative_loss = loss_fn(anchor, negative, target)
        total_loss += negative_loss
        count += 1

    if count == 0:
        # No valid pairs found
        print("No valid positive or negative pairs found.")
        return torch.tensor(0.0, device=device, requires_grad=True)

    average_loss = total_loss / count
    return average_loss

def reconstruction_loss(reconstructed_embeddings, original_embeddings):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # original_embeddings shape: [batch_size, seq_length, embed_dim]
    loss_fn = nn.MSELoss()
    loss = loss_fn(reconstructed_embeddings, original_embeddings)
    return loss

# Loss Function
def total_loss(
    binarized_embeddings_list,
    reconstructed_embeddings_list,
    original_embeddings_list,
    labels_list,
    recon_loss_weight=1,
    semantic_loss_weight=5.0,
    distance_loss_weight=5.0,
    contrastive_loss_weight=1.0,
    diversity_loss_weight=3.0  # Add a weight for diversity loss
):
    total_recon_loss = 0.0
    total_semantic_loss = 0.0
    total_distance_loss = 0.0
    total_contrastive_loss = 0.0
    total_diversity_loss = 0.0  # Initialize diversity loss
    num_bit_lengths = len(binarized_embeddings_list)

    for bin_embeddings, recon_embeddings, orig_embeddings, labels in zip(
        binarized_embeddings_list,
        reconstructed_embeddings_list,
        original_embeddings_list,
        labels_list
    ):
        # Reconstruction Loss
        recon_loss = torch.nn.functional.mse_loss(recon_embeddings, orig_embeddings)
        total_recon_loss += recon_loss

        # Semantic Similarity Loss
        sem_loss = semantic_similarity_loss(bin_embeddings)
        total_semantic_loss += sem_loss

        # Distance Preservation Loss
        dist_loss = distance_preservation_loss(orig_embeddings, bin_embeddings, num_pairs=32)
        total_distance_loss += dist_loss

        # Contrastive Loss
        cont_loss = contrastive_loss(bin_embeddings, labels)
        total_contrastive_loss += cont_loss

        # Diversity Loss
        div_loss = diversity_loss(bin_embeddings.view(-1, bin_embeddings.size(-1)))
        total_diversity_loss += div_loss

    # Averages
    average_recon_loss = total_recon_loss / num_bit_lengths
    average_semantic_loss = total_semantic_loss / num_bit_lengths
    average_distance_loss = total_distance_loss / num_bit_lengths
    average_contrastive_loss = total_contrastive_loss / num_bit_lengths
    average_diversity_loss = total_diversity_loss / num_bit_lengths

    # Total Loss
    loss = (
        recon_loss_weight * average_recon_loss
        + semantic_loss_weight * average_semantic_loss
        + distance_loss_weight * average_distance_loss
        + contrastive_loss_weight * average_contrastive_loss
        + diversity_loss_weight * average_diversity_loss  # Include diversity loss
    )
    return loss, average_recon_loss, average_semantic_loss, average_distance_loss, average_contrastive_loss, average_diversity_loss

MODEL_PARAMS = {
    'vocab_size': vocab_size,
    'embed_dim': 4096,
    'num_heads': 16,
    'hidden_dim': 8192,
    'num_layers': 6,
    'bit_lengths': [1024, 2048, 4096],
    'max_seq_length': 32,
    'dropout': 0.2
}

from scipy.stats import spearmanr

def compute_similarity_scores(embeddings_dict, similarity_df):
    human_scores = []
    model_scores = []
    skipped_pairs = 0

    for _, row in similarity_df.iterrows():
        word1 = row['Word 1']
        word2 = row['Word 2']
        human_score = row['Human (mean)']

        emb1 = embeddings_dict.get(word1)
        emb2 = embeddings_dict.get(word2)

        if emb1 is not None and emb2 is not None:
            # Compute cosine similarity
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            model_scores.append(similarity)
            human_scores.append(human_score)
        else:
            # Skip pairs where embeddings are not available
            skipped_pairs += 1
            continue

    if len(set(model_scores)) <= 1 or len(set(human_scores)) <= 1:
        correlation = 0.0
    else:
        correlation, _ = spearmanr(human_scores, model_scores)
        if math.isnan(correlation):
            correlation = 0.0

    print(f"Skipped {skipped_pairs} word pairs due to missing embeddings.")

    return correlation

def evaluate_analogy_task(embeddings_dict, embeddings_tensor, words_list, analogy_questions, top_k=1):
    correct = 0
    total = 0

    word_to_index = {word: idx for idx, word in enumerate(words_list)}
    embeddings_tensor = nn.functional.normalize(embeddings_tensor, p=2, dim=1)

    for question in analogy_questions:
        word_a, word_b, word_c, word_d = question
        total += 1

        if all(word in word_to_index for word in [word_a, word_b, word_c, word_d]):
            emb_a = embeddings_tensor[word_to_index[word_a]]
            emb_b = embeddings_tensor[word_to_index[word_b]]
            emb_c = embeddings_tensor[word_to_index[word_c]]

            analogy_vector = emb_b - emb_a + emb_c
            analogy_vector = nn.functional.normalize(analogy_vector.unsqueeze(0), p=2, dim=1).squeeze(0)

            similarities = torch.matmul(embeddings_tensor, analogy_vector)
            # Exclude words in the question
            for word in [word_a, word_b, word_c]:
                idx = word_to_index[word]
                similarities[idx] = -float('inf')

            top_indices = similarities.topk(top_k).indices.tolist()
            predicted_words = [words_list[idx] for idx in top_indices]

            if word_d in predicted_words:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# Training Function
def train(rank, world_size):
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    from scipy.stats import spearmanr
    import warnings
    import traceback  # Ensure traceback is imported

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # **Set the device based on rank**
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # **Initialize the process group**
    dist.init_process_group(
        backend='nccl',       # Use 'nccl' for CUDA devices
        init_method='env://', # Ensure env variables are set in main
        world_size=world_size,
        rank=rank
    )

    try:
        # **Set random seed for reproducibility**
        set_seed(42)

        scaler = GradScaler()

        # **Load evaluation datasets (only in the main process)**
        if rank == 0:
            similarity_df = load_simlex999_dataset()
            print("SimLex-999 dataset loaded successfully.")

            analogy_questions = download_google_analogy_dataset()
            print("Google Analogy dataset loaded successfully.")
        else:
            similarity_df = None
            analogy_questions = None

        # **Broadcast the datasets to all processes**
        similarity_list = [similarity_df]
        dist.broadcast_object_list(similarity_list, src=0)
        similarity_df = similarity_list[0]

        analogy_list = [analogy_questions]
        dist.broadcast_object_list(analogy_list, src=0)
        analogy_questions = analogy_list[0]

        # **Load the train and validation datasets**
        train_dataset_path = 'tokenized_dataset_train'
        val_dataset_path = 'tokenized_dataset_val'

        if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path):
            if rank == 0:
                print(f"Process {rank}: Loading train dataset from {train_dataset_path}...")
                print(f"Process {rank}: Loading validation dataset from {val_dataset_path}...")

            train_dataset = load_from_disk(train_dataset_path)
            val_dataset = load_from_disk(val_dataset_path)

            if rank == 0:
                print(f"Process {rank}: Datasets loaded successfully.")
        else:
            raise FileNotFoundError("Train and validation datasets not found.")

        # **Set dataset format for PyTorch**
        train_dataset.set_format(type='torch', columns=['input_ids'])
        val_dataset.set_format(type='torch', columns=['input_ids'])

        # **Create DistributedSampler**
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # **DataLoader**
        batch_size = 1536  # Adjust batch size as needed
        num_workers = 8    # Adjust as needed
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )

        # **Model initialization**
        model = BinaryEmbeddingModel(**MODEL_PARAMS).to(device)
        model.apply(initialize_weights)
        model = DDP(model, device_ids=[rank], output_device=rank)

        # **Optimizer and Scheduler**
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # **Early Stopping Parameters**
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        patience = 5  # Number of epochs with no improvement after which training will be stopped

        num_epochs = 10  # Adjust as needed

        if rank == 0:
            print("Starting training...")

        for epoch in range(num_epochs):
            # **Set the epoch for the sampler**
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            # **Update loss weights**
            semantic_loss_weight = get_semantic_loss_weight(epoch, num_epochs)

            model.train()
            total_epoch_loss = 0

            if rank == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} started.")

            for batch_idx, batch in enumerate(train_dataloader):
                inputs = batch['input_ids'].to(device)
                src_key_padding_mask = (inputs == tokenizer.pad_token_id).to(device)
                labels = inputs.clone()
                labels_list = [labels for _ in MODEL_PARAMS['bit_lengths']]

                # Check for NaN or infinite values in inputs
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"Skipping batch {batch_idx} due to invalid input values.")
                    continue  # Skip this batch

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda'):
                    # Forward pass
                    binarized_embeddings_list, reconstructed_embeddings_list, original_embeddings, labels = model(
                        inputs, src_key_padding_mask=src_key_padding_mask
                    )

                    # Prepare original_embeddings_list
                    original_embeddings_list = [original_embeddings for _ in binarized_embeddings_list]

                    # Compute total loss
                    loss, recon_loss, semantic_loss, distance_loss, contrastive_loss_val, diversity_loss_val = total_loss(
                        binarized_embeddings_list,
                        reconstructed_embeddings_list,
                        original_embeddings_list,
                        labels_list,
                        recon_loss_weight=1,
                        semantic_loss_weight=semantic_loss_weight,
                        distance_loss_weight=5,
                        contrastive_loss_weight=3,
                        diversity_loss_weight=5
                    )
                    # Add these print statements
                    if rank == 0 and batch_idx == 0:
                        print(f"Epoch {epoch}, Batch {batch_idx}")
                        print("Sample binarized embeddings:")
                        print(binarized_embeddings_list[0][0][:10])  # First 10 values of first embedding in first bit length
                        print("Sample reconstructed embeddings:")
                        print(reconstructed_embeddings_list[0][0][:10])  # First 10 values of first reconstructed embedding
                        print("Sample original embeddings:")
                        print(original_embeddings[0][:10])  # First 10 values of first original embedding`


                    if loss is None or not torch.isfinite(loss):
                        continue  # Skip this batch

                # Backpropagation and optimization
                scaler.scale(loss).backward()

                # Unscale gradients before clipping
                scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Logging
                total_epoch_loss += loss.item()

                if rank == 0 and (batch_idx + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                        f"Total Loss: {loss.item():.6f}, "
                        f"Recon Loss: {recon_loss.item():.6f}, "
                        f"Semantic Loss: {semantic_loss.item():.6f}, "
                        f"Distance Loss: {distance_loss.item():.6f}, "
                        f"Contrastive Loss: {contrastive_loss_val.item():.6f}, "
                        f"Diversity Loss: {diversity_loss_val.item():.6f}"
                    )

            avg_epoch_loss = total_epoch_loss / len(train_dataloader)

            if rank == 0:
                print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")

            # **Validation Phase**
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_inputs = val_batch['input_ids'].to(device)
                    val_src_key_padding_mask = (val_inputs == tokenizer.pad_token_id).to(device)

                    # Forward pass
                    val_binarized_embeddings_list, val_reconstructed_embeddings_list, val_original_embeddings, val_labels = model(
                        val_inputs, src_key_padding_mask=val_src_key_padding_mask
                    )

                    # Prepare original_embeddings_list
                    val_original_embeddings_list = [val_original_embeddings for _ in val_binarized_embeddings_list]

                    # Prepare labels list
                    val_labels_list = [val_labels for _ in val_binarized_embeddings_list]

                    # Compute loss
                    val_loss, val_recon_loss, val_semantic_loss, val_distance_loss, val_contrastive_loss, val_diversity_loss = total_loss(
                        val_binarized_embeddings_list,
                        val_reconstructed_embeddings_list,
                        val_original_embeddings_list,
                        val_labels_list,
                        recon_loss_weight=1,
                        semantic_loss_weight=semantic_loss_weight,
                        distance_loss_weight=5,
                        contrastive_loss_weight=3,
                        diversity_loss_weight=5
                    )

                    if torch.isnan(val_loss) or not torch.isfinite(val_loss):
                        continue  # Skip this batch

                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)

            if rank == 0:
                print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

            # **Synchronize before benchmarking**
            dist.barrier()

            # **Benchmarking and Model Saving (only in rank 0)**
            '''if rank == 0:
                # Prepare embeddings for benchmarking
                model.eval()
                with torch.no_grad():
                    # Precompute embeddings using full vocabulary
                    bit_length = MODEL_PARAMS['bit_lengths'][0]  # Use desired bit length
                    model_to_use = model.module if hasattr(model, 'module') else model
                    words_list, embeddings_tensor = precompute_embeddings(
                        model_to_use,
                        tokenizer, bit_length=bit_length, device=device
                    )
                    embeddings_dict = {word: embeddings_tensor[idx] for idx, word in enumerate(words_list)}

                # Word Similarity Benchmark
                correlation = compute_similarity_scores(embeddings_dict, similarity_df)
                print(f"Epoch [{epoch+1}/{num_epochs}], Word Similarity Spearman Correlation: {correlation:.4f}")

                # Analogy Benchmark
                accuracy = evaluate_analogy_task(embeddings_dict, embeddings_tensor, words_list, analogy_questions)
                print(f"Epoch [{epoch+1}/{num_epochs}], Analogy Task Accuracy: {accuracy:.4f}")
                model.train()
'''
                # Learning Rate Scheduler Step
            scheduler.step(avg_val_loss)

            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                # **Save the best model**
                torch.save(model.module.state_dict(), 'best_model.pth')
                print("Validation loss improved. Model saved.")
            else:
                epochs_without_improvement += 1
                print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {patience} epochs with no improvement.")
                    break

            # **Synchronize before next epoch**
            dist.barrier()

        if rank == 0:
            print("Training completed.")

        # **Synchronize all processes before cleanup**
        dist.barrier()

    except Exception as e:
        print(f"Process {rank}: Encountered an exception: {e}")
        traceback.print_exc()
    finally:
        # **Ensure cleanup is called**
        dist.destroy_process_group()

# Function to precompute embeddings for all words in the tokenizer's vocabulary
def precompute_embeddings(model, tokenizer, bit_length, device):
    model.eval()
    vocab = tokenizer.get_vocab()
    words_list = []
    embeddings_list = []
    batch_size = 128

    # Filter out special tokens and [unused tokens
    filtered_vocab = [word for word in vocab.keys() if not word.startswith('[') and not word.startswith('##') and not word.startswith('<')]

    with torch.no_grad():
        for i in range(0, len(filtered_vocab), batch_size):
            batch_words = filtered_vocab[i:i+batch_size]
            batch_ids = tokenizer(batch_words, padding='max_length', truncation=True, max_length=MODEL_PARAMS['max_seq_length'], return_tensors='pt')['input_ids'].to(device)
            
            binarized_embeddings, _, _, _ = model(batch_ids)
            embeddings = binarized_embeddings[0]  # Assuming first bit length
            
            # Ensure all embeddings have the same size
            if embeddings.dim() == 3:  # If embeddings are [batch_size, seq_len, embed_dim]
                embeddings = embeddings.mean(dim=1)  # Average over sequence length
            elif embeddings.dim() == 2:  # If embeddings are already [batch_size, embed_dim]
                pass
            else:
                raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
            
            words_list.extend(batch_words)
            embeddings_list.append(embeddings.cpu())

    # Ensure all tensors in embeddings_list have the same size
    embed_dim = embeddings_list[0].size(1)
    embeddings_list = [emb[:, :embed_dim] for emb in embeddings_list]

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    print(f"Precomputed embeddings: min={embeddings_tensor.min().item():.2f}, max={embeddings_tensor.max().item():.2f}, mean={embeddings_tensor.mean().item():.2f}")
    return words_list, embeddings_tensor

# Function to compute analogies
def compute_analogy(word_a, word_b, word_c, embeddings_dict):
    # Retrieve embeddings for the words
    emb_a = embeddings_dict.get(word_a)
    emb_b = embeddings_dict.get(word_b)
    emb_c = embeddings_dict.get(word_c)
    if emb_a is None or emb_b is None or emb_c is None:
        print("One of the words is not in the vocabulary.")
        return None
    analogy_vector = emb_a - emb_b + emb_c
    return analogy_vector

# Finding the most similar words to a given embedding
def find_most_similar_embedding(embedding, embeddings_tensor, words_list, top_k=5, exclude_words=None):
    import torch
    import re

    if exclude_words is None:
        exclude_words = set()
    else:
        exclude_words = set(exclude_words)

    # Ensure tensors are on the same device
    embedding = embedding.to(embeddings_tensor.device)

    # Ensure embedding is 2D
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)

    # embeddings_tensor should be 2D
    if embeddings_tensor.dim() > 2:
        embeddings_tensor = embeddings_tensor.view(embeddings_tensor.size(0), -1)

    # Exclude [unusedX] tokens and non-alphabetic words with at least 2 letters
    pattern = re.compile("^[a-zA-Z]{2,}$")
    filtered_indices = [
        i for i, word in enumerate(words_list)
        if pattern.match(word) and word not in exclude_words
    ]
    filtered_embeddings = embeddings_tensor[filtered_indices]
    filtered_words = [words_list[i] for i in filtered_indices]

    # Compute similarities using cosine similarity
    similarities = torch.nn.functional.cosine_similarity(embedding, filtered_embeddings)

    # Get top_k most similar embeddings
    top_k = min(top_k, len(filtered_words))
    similarities, indices = torch.topk(similarities, top_k)

    indices = indices.cpu().numpy()

    top_words = [(filtered_words[int(idx)], similarities[i].item()) for i, idx in enumerate(indices)]
    return top_words

# Function to find most similar words to a given word
def find_most_similar_words(target_word, embeddings_dict, embeddings_tensor, words_list, top_k=15):
    if target_word not in embeddings_dict:
        print(f"Word '{target_word}' not found in the embeddings.")
        return None

    target_embedding = embeddings_dict[target_word].unsqueeze(0)  # Shape: [1, embed_dim]

    # Normalize embeddings
    embeddings_tensor = nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    target_embedding = nn.functional.normalize(target_embedding, p=2, dim=1)

    # Compute cosine similarities
    similarities = torch.matmul(embeddings_tensor, target_embedding.T).squeeze(1)  # Shape: [num_words]

    # Get top_k most similar words
    top_k_indices = similarities.topk(top_k + 1).indices.tolist()  # +1 to exclude the target word itself
    top_words = []
    for idx in top_k_indices:
        word = words_list[idx]
        if word != target_word:
            similarity = similarities[idx].item()
            top_words.append((word, similarity))
            if len(top_words) == top_k:
                break

    return top_words

# Main function
def main():
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    from collections import OrderedDict

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12455'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    import warnings
    from transformers import logging as hf_logging

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Suppress warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    hf_logging.set_verbosity_error()

    # Prepare datasets
    print("Preparing datasets...")
    prepare_datasets()
    
    # Load evaluation datasets
    similarity_df = load_simlex999_dataset()
    print("SimLex-999 dataset loaded successfully.")
    analogy_questions = download_google_analogy_dataset()

    # Check if 'best_model.pth' exists
    if os.path.exists('best_model.pth'):
        print("Found existing model. Skipping training.")
    else:
        print("No saved model found. Starting training from scratch.")
        # Determine the number of GPUs available
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No CUDA devices available.")

        # Spawn processes for training
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

    # Load the trained model
    model = BinaryEmbeddingModel(**MODEL_PARAMS).to(device)
    print("Loading trained model from 'best_model.pth'...")
    state_dict = torch.load('best_model.pth', map_location=device)

    # Remove 'module.' prefix from the keys in state_dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    # Add these print statements
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: min={param.min().item():.2f}, max={param.max().item():.2f}, mean={param.mean().item():.2f}")

    model.eval()

    # Precompute embeddings
    bit_length = MODEL_PARAMS['bit_lengths'][0]
    print("Precomputing embeddings...")
    words_list, embeddings_tensor = precompute_embeddings(model, tokenizer, bit_length=bit_length, device=device)

    # Create embeddings dictionary
    embeddings_dict = {word: embeddings_tensor[idx] for idx, word in enumerate(words_list)}

    # Save embeddings
    torch.save({
        'embeddings_tensor': embeddings_tensor,
        'words_list': words_list
    }, 'binary_embeddings.pth')
    print("Binary embeddings saved to 'binary_embeddings.pth'.")

    # Perform analogy task
    word_a, word_b, word_c = 'king', 'man', 'woman'
    analogy_vector = compute_analogy(word_a, word_b, word_c, embeddings_dict)

    if analogy_vector is not None:
        top_words = find_most_similar_embedding(analogy_vector, embeddings_tensor, words_list, top_k=15)
        print(f"\nTop words for the analogy '{word_a} - {word_b} + {word_c}':")
        for word, similarity in top_words:
            print(f"{word}: Similarity = {similarity:.4f}")

    # Find similar words
    target_word = 'goat'
    similar_words = find_most_similar_words(target_word, embeddings_dict, embeddings_tensor, words_list, top_k=15)
    if similar_words is not None:
        print(f"\nMost similar words to '{target_word}':")
        for word, similarity in similar_words:
            print(f"{word}: Similarity = {similarity:.4f}")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method(method='spawn')
    main()
'''
def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryEmbeddingModel(**MODEL_PARAMS).to(device)
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (32, 16)).to(device)
    src_key_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    binarized_embeddings, reconstructed_embeddings, original_embeddings, labels = model(input_ids, src_key_padding_mask)
    
    print("Forward pass successful")
    
    # Test loss computation
    loss, recon_loss, semantic_loss, distance_loss, contrastive_loss_val, diversity_loss_val = total_loss(
        binarized_embeddings,
        reconstructed_embeddings,
        [original_embeddings for _ in binarized_embeddings],
        [labels for _ in binarized_embeddings],
        recon_loss_weight=1,
        semantic_loss_weight=1,
        distance_loss_weight=5,
        contrastive_loss_weight=1,
        diversity_loss_weight=3
    )
    
    print("Loss computation successful")
    print(f"Total loss: {loss.item()}")
    
    # Test backward pass
    loss.backward()
    print("Backward pass successful")

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method(method='spawn')
    test_model()
    '''