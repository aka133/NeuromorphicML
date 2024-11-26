# NeuromorphicML

## Project Overview
This repository documents the evolution of a sophisticated neuromorphic computing system, progressing from basic distributed neural networks to advanced spiking neural architectures with bio-inspired mechanisms. The research spans embedding compression, spiking neural networks, and biologically-inspired computing architectures.

## Project Evolution & Technical Components

### 1. Initial Neural Network Architecture (`5am.py`)
My initial implementation focused on distributed training fundamentals. The system utilizes PyTorch's DistributedDataParallel for efficient multi-GPU training, incorporating a sophisticated analogy evaluation system for word embeddings. I implemented comprehensive benchmarking for word similarity and analogy tasks, along with early stopping and model checkpointing for optimal training outcomes.

### 2. Spiking Neural Network Integration (`spikformer.py`)
Building on the distributed foundation, I developed a novel hybrid architecture combining transformers with spiking neural networks. The system features custom spike encoding/decoding mechanisms specifically designed for text processing, alongside a phase-shifted spiking self-attention mechanism. I integrated population coding for neural representation and developed sophisticated temporal processing with positional encoding.

### 3. Population Coding & Temporal Processing (`popspike.py`)
This component represents a significant advance in neural processing, featuring sophisticated population coding mechanisms for word embeddings. The system implements temporal sequence processing with working memory integration, complemented by efficient binary embedding systems for storage optimization. I built comprehensive benchmarking systems for both spike-based and continuous embeddings, utilizing FAISS indexing for efficient similarity search.

### 4. Matryoshka Binary Embedder (`matryoshka_binary_embedder.py`)
This innovative system implements hierarchical binary embeddings using Matryoshka Representation Learning, starting with continuous embeddings and converting them to efficient binary representations supporting multiple resolutions from 1024 to 8192 bits. The architecture employs straight-through estimators for binary gradients and enables hierarchical reconstruction capabilities. The system features a multi-component loss function balancing reconstruction quality with semantic similarity preservation, alongside comprehensive evaluation metrics including word similarity benchmarking and analogy task evaluation.

### 5. Advanced Spiking Architecture (`spikingjam2.py`)
My latest spiking neural network architecture achieves 100% reconstruction accuracy through a combination of rate coding and population coding, successfully converting continuous embeddings into neural spikes. The system implements sophisticated population coding with temporal position encoding and rate-based spike generation. The system features phase-shifted spiking attention mechanisms and leaky integrate-and-fire neurons, managed through careful membrane potential control systems.

### 6. Advanced Neural Architecture (`cerebro8.py`)
My most sophisticated neural simulation environment implements a multi-layer cortical architecture with integrated working memory systems. The architecture features sophisticated visualization tools for network analysis and implements Spike-Timing-Dependent Plasticity (STDP) for biologically-inspired learning processes.

## Technical Achievements & Innovations

### Embedding Compression Pipeline
Starting with continuous embeddings from FastText and OpenAI's ada-02, I developed a binary autoencoder using sigma-delta quantization to preserve Euclidean distances. The system progressed to handle concatenated embedding models and ultimately achieved a novel compression using Microsoft's 1.58 bit LLM compression technique, resulting in 10,000x compression savings while maintaining 100% reconstruction accuracy.

### Distributed Computing
I implemented comprehensive multi-GPU training support with efficient data parallel processing pipelines. My streaming dataset processing and distributed sampler implementation enable scalable training across multiple computing nodes.

### Neural Architecture Design
Key innovations include the fusion of traditional transformers with spiking neural networks, implementation of novel population coding mechanisms, and development of phase-shifted attention mechanisms. The architecture maintains biological plausibility while optimizing for computational efficiency.

### Performance Optimization
My binary embedding systems achieve an 87.5% reduction in memory usage when converting from continuous to binary representations (8192 to 1024 bits). The integration of spiking networks significantly reduces computational complexity, while my distributed training approach enables unprecedented scalability.

## Project Progression Highlights
The research evolved from a basic neuroevolution model with just 3 neurons to increasingly complex implementations. Key milestones include tokenizing an OpenWebText corpus, converting continuous embeddings to binary representations using 8 H100 GPUs, and implementing sophisticated compression techniques. The project culminated in developing the largest known Liquid State Machine, featuring cortical layers and a midbrain reservoir for biologically realistic organization. I also created a teacher-student distillation framework for automated training of the LSM.

## Technologies & Dependencies
- PyTorch with Distributed Computing Support
- CUDA for GPU Optimization
- Hugging Face Transformers
- FAISS for Similarity Search
- SNNTorch for Spiking Neural Networks
- Scientific Computing Libraries (NumPy, SciPy)
- Visualization Tools (Matplotlib)

## Contact
archit.kalra@rice.edu
