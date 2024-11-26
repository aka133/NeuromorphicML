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

def get_semantic_loss_weight(epoch, total_epochs, initial_weight=0.1, final_weight=1.0):
    return initial_weight * (final_weight / initial_weight) ** (epoch / total_epochs)

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DDP setup and cleanup functions
def setup(rank, world_size):
    import torch.distributed as dist  # Import inside the function
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' if 'nccl' is not supported
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    import torch.distributed as dist  # Import inside the function
    dist.destroy_process_group()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
vocab_size = len(tokenizer)

# Model weight initialization
def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

# Dataset preparation function
def prepare_datasets():
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
                    clean_up_tokenization_spaces=True
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
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Approximate gradient with tanh derivative
        grad_input = grad_output * (1 - torch.tanh(input) ** 2)
        return grad_input

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

        # Return binarized embeddings, reconstructed embeddings, and original embeddings
        return binarized_embeddings, reconstructed_embeddings, x

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

def semantic_similarity_loss(embeddings, context_window=5):
    # embeddings: [batch_size, seq_length, embed_dim]
    batch_size, seq_length, embed_dim = embeddings.size()
    loss_fn = nn.CosineEmbeddingLoss()

    total_loss = 0
    count = 0

    for i in range(seq_length):
        anchor = embeddings[:, i, :]  # Shape: [batch_size, embed_dim]

        # Positive pairs within the context window
        for j in range(max(0, i - context_window), min(seq_length, i + context_window + 1)):
            if i == j:
                continue
            positive = embeddings[:, j, :]
            target = torch.ones(batch_size).to(embeddings.device)  # Similarity target: 1
            positive_loss = loss_fn(anchor, positive, target)
            total_loss += positive_loss
            count += 1

        # Negative pairs with random words in the batch
        neg_indices = torch.randint(0, seq_length, (batch_size,)).to(embeddings.device)
        negative = embeddings[torch.arange(batch_size), neg_indices, :]
        target = -torch.ones(batch_size).to(embeddings.device)  # Similarity target: -1
        negative_loss = loss_fn(anchor, negative, target)
        total_loss += negative_loss
        count += 1

    average_loss = total_loss / count
    return average_loss

def reconstruction_loss(reconstructed_embeddings, original_embeddings):
    losses = []
    for z_reconstructed in reconstructed_embeddings:
        loss = nn.MSELoss()(z_reconstructed, original_embeddings)
        losses.append(loss)
    # You can sum or average the losses
    total_loss = sum(losses) / len(losses)
    return total_loss

# Loss Function
def total_loss(
    binarized_embeddings, reconstructed_embeddings, original_embeddings,
    semantic_loss_weight=1.0  # Default value if not provided
):
    # Reconstruction Loss
    recon_loss = reconstruction_loss(reconstructed_embeddings, original_embeddings)
    
    # Semantic Similarity Loss
    semantic_loss = semantic_similarity_loss(original_embeddings)
    
    # Total Loss - Combine losses with dynamic weighting
    loss = recon_loss + semantic_loss_weight * semantic_loss
    return loss, recon_loss, semantic_loss

MODEL_PARAMS = {
    'vocab_size': vocab_size,
    'embed_dim': 4096,
    'num_heads': 8,
    'hidden_dim': 8192,
    'num_layers': 6,
    'bit_lengths': [1024, 2048, 4096, 8192],
    'max_seq_length': 32,
    'dropout': 0.2
}

from scipy.stats import spearmanr

def compute_similarity_scores(embeddings_dict, similarity_df):
    actual_similarities = []
    predicted_similarities = []

    for _, row in similarity_df.iterrows():
        word1, word2, human_score = row['Word 1'], row['Word 2'], row['Human (mean)']
        word1, word2 = word1.lower(), word2.lower()

        if word1 in embeddings_dict and word2 in embeddings_dict:
            emb1 = embeddings_dict[word1]
            emb2 = embeddings_dict[word2]

            print(f"emb1 shape: {emb1.shape}, emb2 shape: {emb2.shape}")

            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1
            ).item()

            actual_similarities.append(human_score)
            predicted_similarities.append(cos_sim)

    # Compute Spearman correlation
    if len(actual_similarities) > 0:
        correlation, _ = spearmanr(actual_similarities, predicted_similarities)
    else:
        correlation = 0

    return correlation

def evaluate_analogy_task(embeddings_dict, embeddings_tensor, words_list, analogy_questions):
    correct = 0
    total = 0

    # Create a set for faster lookup
    vocab_set = set(words_list)

    for a, b, c, d in analogy_questions:
        if all(word in vocab_set for word in [a, b, c, d]):
            emb_a = embeddings_dict[a]
            emb_b = embeddings_dict[b]
            emb_c = embeddings_dict[c]

            # Compute analogy vector: b - a + c
            analogy_vector = emb_b - emb_a + emb_c

            # Find the most similar words to the analogy vector
            top_words = find_most_similar_embedding(
                analogy_vector, embeddings_tensor, words_list, top_k=5, exclude_words=[a, b, c]
            )
            predicted_words = [word for word, _ in top_words]

            if d in predicted_words:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Training Function
def train(rank, world_size):
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist  # Import if 'dist' is used
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    from scipy.stats import spearmanr
    
    similarity_df = load_simlex999_dataset()
    print("SimLex-999 dataset loaded successfully.")

    analogy_questions = download_google_analogy_dataset()
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer)

    try:
        setup(rank, world_size)
        set_seed(42)
        device = torch.device(f'cuda:{rank}')

        # Load the train and validation datasets
        train_dataset_path = 'tokenized_dataset_train'
        val_dataset_path = 'tokenized_dataset_val'

        if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path):
            if rank == 0:
                print(f"Process {rank}: Loading train dataset from {train_dataset_path}...")
                print(f"Process {rank}: Loading validation dataset from {val_dataset_path}...")
            train_dataset = load_from_disk(train_dataset_path)
            val_dataset = load_from_disk(val_dataset_path)
            if rank == 0:
                print(f"Process {rank}: Train and validation datasets loaded.")
        else:
            raise FileNotFoundError(
                f"Train and validation datasets not found at {train_dataset_path} and {val_dataset_path}. Please run 'prepare_datasets()' first."
            )

        # Since we're using DistributedSampler, we need to set the format for PyTorch tensors
        train_dataset.set_format(type='torch', columns=['input_ids'])
        val_dataset.set_format(type='torch', columns=['input_ids'])

        # Create DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        # DataLoader
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

        # Model initialization
        model = BinaryEmbeddingModel(**MODEL_PARAMS).to(device)
        model.apply(initialize_weights)
        model = DDP(model, device_ids=[rank])

        # Optimizer and Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Mixed precision
        scaler = torch.amp.GradScaler()

        # Early Stopping Parameters
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        patience = 5  # Number of epochs with no improvement after which training will be stopped

        num_epochs = 20  # Adjust as needed

        if rank == 0:
            print("Starting training...")

        for epoch in range(num_epochs):
            semantic_loss_weight = get_semantic_loss_weight(epoch, num_epochs)

            train_sampler.set_epoch(epoch)
            model.train()
            total_epoch_loss = 0

            if rank == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} started.")

            for batch_idx, batch in enumerate(train_dataloader):
                # Data preparation
                inputs = batch['input_ids'].to(device)
                src_key_padding_mask = (inputs == tokenizer.pad_token_id).to(device)

                # Use autocast with required arguments
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    binarized_embeddings, reconstructed_embeddings, original_embeddings = model(
                        inputs, src_key_padding_mask=src_key_padding_mask
                    )

                    # Compute total loss
                    loss, recon_loss, semantic_loss = total_loss(
                        binarized_embeddings, reconstructed_embeddings, original_embeddings,
                        semantic_loss_weight=semantic_loss_weight  # Pass the dynamic weight
                    )
                    if loss is None or not torch.isfinite(loss):
                        continue  # Skip this batch

                # Backpropagation and optimization
                scaler.scale(loss).backward()

                # Optional gradient clipping (after unscaling)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_epoch_loss += loss.item()

                if rank == 0 and (batch_idx + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                        f"Total Loss: {loss.item():.6f}, "
                        f"Recon Loss: {recon_loss.item():.6f}, "
                        f"Semantic Loss: {semantic_loss.item():.6f}"
                    )

            avg_epoch_loss = total_epoch_loss / len(train_dataloader)

            if rank == 0:
                print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")

            # Validation Phase
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_inputs = val_batch['input_ids'].to(device)
                    val_src_key_padding_mask = (val_inputs == tokenizer.pad_token_id).to(device)

                    # Forward pass
                    val_binarized_embeddings, val_reconstructed_embeddings, val_original_embeddings = model(val_inputs, src_key_padding_mask=val_src_key_padding_mask)

                    # Compute loss
                    # Validation loop snippet (modified)
                    val_loss, val_recon_loss, val_semantic_loss = total_loss(
                        val_binarized_embeddings, val_reconstructed_embeddings, val_original_embeddings
                    )

                    if torch.isnan(val_loss) or not torch.isfinite(val_loss):
                        continue  # Skip this batch

                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)

            if rank == 0:
                print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

            if rank == 0:
                # Prepare embeddings for benchmarking
                model.eval()
                with torch.no_grad():
                    # Precompute embeddings
                    bit_length = MODEL_PARAMS['bit_lengths'][0]  # Use the smallest bit length
                    words_list, embeddings_tensor = precompute_embeddings(model.module, tokenizer, vocab_size, bit_length=bit_length, device=device)
                    embeddings_dict = {word: embeddings_tensor[idx] for idx, word in enumerate(words_list)}
                
                # Word Similarity Benchmark
                correlation = compute_similarity_scores(embeddings_dict, similarity_df)
                print(f"Epoch [{epoch+1}/{num_epochs}], Word Similarity Spearman Correlation: {correlation:.4f}")

                # Analogy Benchmark
                accuracy = evaluate_analogy_task(embeddings_dict, embeddings_tensor, words_list, analogy_questions)
                print(f"Epoch [{epoch+1}/{num_epochs}], Analogy Task Accuracy: {accuracy:.4f}")

                model.train()

            # Learning Rate Scheduler Step
            scheduler.step(avg_val_loss)

            # Early Stopping Check
            if rank == 0:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    # Save the best model
                    torch.save(model.module.state_dict(), 'best_model.pth')
                    print("Validation loss improved. Model saved.")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered after {patience} epochs with no improvement.")
                        break

            # Save the model after each epoch or when validation loss improves
            if rank == 0:
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Model saved at epoch {epoch+1}")

        if rank == 0:
            print("Training completed.")

    except Exception as e:
        print(f"Exception in process {rank}: {e}")
        traceback.print_exc()
    finally:
        cleanup()

# Function to precompute embeddings
def precompute_embeddings(model, tokenizer, vocab_size, bit_length, device):
    model.eval()
    words_list = []
    embeddings_list = []

    with torch.no_grad():
        for idx in range(vocab_size):
            word = tokenizer.convert_ids_to_tokens(idx)
            if word in tokenizer.all_special_tokens:
                continue  # Skip special tokens

            words_list.append(word)
            token_tensor = torch.tensor([[idx]]).to(device)

            # Since we're processing single tokens, src_key_padding_mask is not needed
            src_key_padding_mask = None

            # Get the binarized embedding
            binarized_embeddings, reconstructed_embeddings, _ = model(
                token_tensor, src_key_padding_mask=src_key_padding_mask
            )

            # Find the index of the desired bit length
            bit_length_index = model.bit_lengths.index(bit_length)
            bin_embedding = binarized_embeddings[bit_length_index].squeeze(0).cpu()
            embeddings_list.append(bin_embedding)

    embeddings_tensor = torch.stack(embeddings_list)
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
    import torch
    import re

    if target_word not in embeddings_dict:
        print(f"Word '{target_word}' not found in the embeddings.")
        return None

    # Get embedding for the target word
    emb = embeddings_dict[target_word]

    # Ensure embedding is 2D
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)

    # embeddings_tensor should be 2D
    if embeddings_tensor.dim() > 2:
        embeddings_tensor = embeddings_tensor.view(embeddings_tensor.size(0), -1)

    # **Exclude [unusedX] tokens and non-alphanumeric tokens**
    pattern = re.compile('^[a-zA-Z0-9]+$')
    filtered_indices = [
        i for i, word in enumerate(words_list)
        if not word.startswith('[unused') and pattern.match(word)
    ]
    filtered_embeddings = embeddings_tensor[filtered_indices]
    filtered_words = [words_list[i] for i in filtered_indices]

    # Compute similarities
    similarities = torch.nn.functional.cosine_similarity(emb, filtered_embeddings)

    # Exclude the target word itself (if present in filtered_words)
    if target_word in filtered_words:
        target_idx = filtered_words.index(target_word)
        similarities[target_idx] = float('-inf')

    # Get top_k most similar words
    top_k = min(top_k, len(filtered_words))
    similarities, indices = torch.topk(similarities, top_k)

    # **Convert indices to integers**
    indices = indices.cpu().numpy()

    top_words = [
        (filtered_words[int(idx)], similarities[i].item()) for i, idx in enumerate(indices)
    ]
    return top_words

# Main function
def main():
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    import warnings
    from transformers import logging as hf_logging

    # Suppress FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Suppress specific transformers warnings
    hf_logging.set_verbosity_error()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # Prepare datasets in the main process
    print("Preparing datasets...")
    prepare_datasets()
    
    # Load the dataset
    similarity_df = load_simlex999_dataset()
    print("SimLex-999 dataset loaded successfully.")
    print(similarity_df.head())

    analogy_questions = download_google_analogy_dataset()

    # Check if 'best_model.pth' exists
    if os.path.exists('best_model.pth'):
        print("Loading model from 'best_model.pth'...")
        # Initialize the model
        model = BinaryEmbeddingModel(**MODEL_PARAMS).to(device)

        state_dict = torch.load('best_model.pth', map_location=device)

        # Remove 'module.' prefix from the keys in state_dict
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k
            new_state_dict[new_key] = v

        # Load the modified state_dict into the model
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
    else:
        print("No saved model found. Starting training from scratch.")

        world_size = torch.cuda.device_count()
        mp.spawn(
            train,
            args=(world_size,),  # Pass only world_size
            nprocs=world_size,
            join=True
        )

        # After training, initialize and load the model
        model = BinaryEmbeddingModel(**MODEL_PARAMS).to(device)

        print("Loading trained model from 'best_model.pth'...")
        state_dict = torch.load('best_model.pth', map_location=device)

        # Remove 'module.' prefix from the keys in state_dict
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        print("Model trained and loaded successfully.")

    model.eval()

    # Specify the bit length for which you want to compute embeddings
    bit_length = 1024

    print("Precomputing embeddings...")
    words_list, embeddings_tensor = precompute_embeddings(model, tokenizer, vocab_size, bit_length=bit_length, device=device)

    # Creating a mapping from words to embeddings
    embeddings_dict = {word: embeddings_tensor[idx] for idx, word in enumerate(words_list)}

    # Save embeddings
    torch.save({
        'embeddings_tensor': embeddings_tensor,
        'words_list': words_list
    }, 'binary_embeddings.pth')
    print("Binary embeddings saved to 'binary_embeddings.pth'.")

    # Computing the analogy 'king' - 'man' + 'woman'
    word_a, word_b, word_c = 'king', 'man', 'woman'
    analogy_vector = compute_analogy(word_a, word_b, word_c, embeddings_dict)

    # Finding the top 5 words closest to the analogy vector
    if analogy_vector is not None:
        top_words = find_most_similar_embedding(analogy_vector, embeddings_tensor, words_list, top_k=5)
        print(f"\nTop 5 words for the analogy '{word_a} - {word_b} + {word_c}':")
        for word, similarity in top_words:
            print(f"{word}: Similarity = {similarity:.4f}")

    # Finding the 15 most similar words to 'goat'
    target_word = 'goat'
    similar_words = find_most_similar_words(target_word, embeddings_dict, embeddings_tensor, words_list, top_k=15)
    if similar_words is not None:
        print(f"\n15 most similar words to '{target_word}':")
        for word, similarity in similar_words:
            print(f"{word}: Similarity = {similarity:.4f}")

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method(method='spawn')
    main()