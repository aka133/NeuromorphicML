import os
import numpy as np
from tqdm import tqdm
import gzip
from datasets import load_dataset
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from scipy.spatial.distance import cosine  # Added this import
from torch.multiprocessing import spawn

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def softmax(x):
    return F.softmax(torch.tensor(x, device='cuda'), dim=-1).cpu().numpy()

def quantize_to_bits(value, bits=5):
    max_val = 2**bits - 1
    return int(round(value * max_val))

def value_to_binary(value, bits=5):
    return format(value, f'0{bits}b')

def continuous_to_binary(embeddings, bits_per_dim=5):
    # L2 normalize embeddings
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    # Scale to [0, 1]
    min_val = norm_embeddings.min(dim=1, keepdim=True)[0]
    max_val = norm_embeddings.max(dim=1, keepdim=True)[0]
    scaled_embeddings = (norm_embeddings - min_val) / (max_val - min_val + 1e-8)
    # Quantize
    scaled_embeddings = scaled_embeddings * (2 ** bits_per_dim - 1)
    quantized_embeddings = scaled_embeddings.round().long()
    # Convert to binary strings
    binary_embeddings = [''.join(format(val.item(), f'0{bits_per_dim}b') for val in emb) for emb in quantized_embeddings]
    return binary_embeddings

def process_batch(batch_data, tokenizer, model, device):
    sentences = batch_data['sentence']
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        is_split_into_words=False  # Ensure tokens are not pre-split
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    batch_word_embeddings = []
    for i in range(len(sentences)):
        tokens_emb = outputs.last_hidden_state[i]  # Shape: [seq_len, hidden_size]
        word_ids = inputs.word_ids(batch_index=i)  # Mapping from token index to word index
        word_embeddings = {}
        word_texts = {}
        
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # Skip special tokens
            if word_id not in word_embeddings:
                word_embeddings[word_id] = []
                # Retrieve the substring corresponding to the word
                start, end = inputs['offset_mapping'][i][token_idx]
                word_text = sentences[i][start:end]
                word_texts[word_id] = word_text
            word_embeddings[word_id].append(tokens_emb[token_idx])
        
        for word_id, embeddings in word_embeddings.items():
            word_embedding = torch.mean(torch.stack(embeddings), dim=0)
            word = word_texts[word_id]
            batch_word_embeddings.append((word, word_embedding))
    
    words_list = [item[0] for item in batch_word_embeddings]
    word_embeddings = torch.stack([item[1] for item in batch_word_embeddings])
    binary_embeddings = continuous_to_binary(word_embeddings)
    return list(zip(words_list, binary_embeddings))

def process_streamed_embeddings(rank, world_size, output_path, batch_size=128, bits_per_dim=5):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation.")

    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    dataset = load_dataset("glue", "sst2", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    model.eval()
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
    print(f"Processing embeddings on GPU {rank} ({torch.cuda.get_device_name(rank)})")
    
    scaler = GradScaler()
    local_results = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for batch in tqdm(dataloader, desc=f"Processing batches on GPU {rank}", disable=rank!=0):
        try:
            with autocast(device_type='cuda', dtype=torch.float16):
                results = process_batch(batch, tokenizer, model.module, device)
            local_results.extend(results)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: ran out of memory on GPU {rank}")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

    end_event.record()
    torch.cuda.synchronize()
    print(f"GPU {rank} processing time: {start_event.elapsed_time(end_event) / 1000:.2f} seconds")
    
    # Gather results from all processes
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, local_results)
    
    if rank == 0:
        # Write results to file
        with gzip.open(output_path, 'wt') as f:
            for results in all_results:
                for word, binary_emb in results:
                    f.write(f"{word} {binary_emb}\n")
        print(f"Binary embeddings saved to {output_path}")
    
    cleanup()

def load_binary_embeddings(file_path):
    embeddings = {}
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            word, embedding = line.strip().split(' ')
            embeddings[word] = embedding
    return embeddings

def hamming_distance(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def binary_to_continuous(binary_embedding, bits_per_dim=5):
    embedding = []
    for i in range(0, len(binary_embedding), bits_per_dim):
        binary_value = binary_embedding[i:i+bits_per_dim]
        value = int(binary_value, 2) / (2**bits_per_dim - 1)
        embedding.append(value)
    return np.array(embedding)

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def get_similar_words_binary(word, embeddings, top_n=5):
    if word not in embeddings:
        return []
    word_embedding = embeddings[word]
    similarities = [(w, 1 - hamming_distance(word_embedding, emb) / len(word_embedding))
                    for w, emb in embeddings.items() if w != word]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def get_similar_words_continuous(word, embeddings, top_n=5):
    if word not in embeddings:
        return []
    word_embedding = binary_to_continuous(embeddings[word])
    similarities = [(w, cosine_similarity(word_embedding, binary_to_continuous(emb)))
                    for w, emb in embeddings.items() if w != word]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def word_analogy_binary(word1, word2, word3, embeddings):
    if word1 not in embeddings or word2 not in embeddings or word3 not in embeddings:
        return None
    emb1, emb2, emb3 = [binary_to_continuous(embeddings[w]) for w in (word1, word2, word3)]
    target_emb = emb2 - emb1 + emb3
    similarities = [(w, cosine_similarity(target_emb, binary_to_continuous(emb)))
                    for w, emb in embeddings.items() if w not in (word1, word2, word3)]
    return max(similarities, key=lambda x: x[1])[0]

if __name__ == "__main__":
    output_path = 'binary_embeddings.txt.gz'
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA GPUs available")
    print(f"Using {world_size} GPUs")
    spawn(process_streamed_embeddings, args=(world_size, output_path), nprocs=world_size)

    # Load embeddings
    embeddings = load_binary_embeddings(output_path)

    # Example usage
    print("Similar words (binary):")
    print(get_similar_words_binary("good", embeddings))

    print("\nSimilar words (continuous):")
    print(get_similar_words_continuous("good", embeddings))

    print("\nWord analogy:")
    print(word_analogy_binary("king", "man", "woman", embeddings))