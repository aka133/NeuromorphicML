# NeuromorphicML

## Project Overview
This repository documents the evolution of a neuromorphic computing system, progressing from basic distributed neural networks to advanced spiking neural architectures with bio-inspired mechanisms. The research spans embedding compression, spiking neural networks, and biologically-inspired computing architectures.

## Project Evolution & Technical Components

### 1. Initial Neural Network Architecture (`5am.py`)
The initial implementation focused on distributed training fundamentals. The system utilizes PyTorch's DistributedDataParallel for efficient multi-GPU training, incorporating an analogy evaluation system for word embeddings. We implemented benchmarking for word similarity and analogy tasks, along with early stopping and model checkpointing for optimal training outcomes.

### 2. Spiking Neural Network Integration (`spikformer.py`)
Building on the distributed foundation, we developed a novel hybrid architecture combining transformers with spiking neural networks. The system features custom spike encoding/decoding mechanisms specifically designed for text processing, alongside a phase-shifted spiking self-attention mechanism. We integrated population coding for neural representation and developed temporal processing with positional encoding.

### 3. Population Coding & Temporal Processing (`popspike.py`)
This component represents an advance in neural processing, featuring population coding mechanisms for word embeddings. The system implements temporal sequence processing with working memory integration, complemented by efficient binary embedding systems for storage optimization. We built benchmarking systems for both spike-based and continuous embeddings, utilizing FAISS indexing for efficient similarity search.

### 4. Word2Vec Binarization System (`word2vecbinarizer.py`)
We developed a distributed binary embedding system that efficiently converts continuous word embeddings to binary representations. Using a 5-bit quantization scheme with L2 normalization, the system preserves semantic relationships while significantly reducing memory requirements. The implementation leverages PyTorch's DistributedDataParallel for multi-GPU processing, incorporating mixed-precision training and automated GPU memory management. The system supports both Hamming distance and cosine similarity comparisons, enabling efficient word analogy tasks in binary space.

### 5. Signal Quantization System (quantizer.py)
We developed a sophisticated signal quantization system implementing multiple orders of Sigma-Delta quantization for optimal signal-to-noise performance. The system features four distinct quantization methods: a greedy one-bit quantizer, and first, second, and third-order Sigma-Delta quantizers. Each implementation progressively improves noise shaping and signal preservation. The third-order system, using carefully tuned coefficients (175/144, -25/108, 7/432), achieves the highest precision quantization through triple-stage integration. The architecture employs Hadamard matrices for signal transformation and implements comprehensive error analysis, including Johnson-Lindenstrauss embedding comparison.

### 6. Neural Evolution Model (`pre-mel3.py`)
Inspired by DeepMind's MuZero and reinforcement learning approaches to brain-like architectures, we developed an evolvable spiking neural network that mimics biological learning processes. The system implements biologically-plausible Leaky Integrate-and-Fire neurons with configurable time constants and adaptive thresholding. The network features dynamic topology adaptation through activity-based pruning and distance-based connectivity rules. Learning mechanisms include Spike-Timing-Dependent Plasticity (STDP), synaptic scaling, and homeostatic plasticity. The architecture demonstrates the potential for reinforcement learning to train brain-like neural networks through evolutionary processes.

### 7. Spike Encoding System (`spikecoder.py`)
I implemented a neural encoding system that converts text into biologically-inspired spike trains. The system utilizes high-density population coding with up to 2500 neurons per dimension and configurable firing rates up to 1000Hz. Using BERT embeddings as initial representations, the implementation maintains semantic relationships through Gaussian tuning curves and temporal dynamics. The spike generation process incorporates distance-based firing rate modulation across 100 time steps, with robust decoding mechanisms for accurate reconstruction. This system represents an advancement in neural coding for language processing, bridging theoretical neuroscience with practical NLP applications.

### 8. Matryoshka Binary Embedder (`matryoshka_binary_embedder.py`)
This system implements hierarchical binary embeddings using Matryoshka Representation Learning, starting with continuous embeddings and converting them to efficient binary representations supporting multiple resolutions from 1024 to 8192 bits. The architecture employs straight-through estimators for binary gradients and enables hierarchical reconstruction capabilities. The system features a multi-component loss function balancing reconstruction quality with semantic similarity preservation, alongside evaluation metrics including word similarity benchmarking and analogy task evaluation.

### 9. Advanced Spiking Architecture (`spikingjam2.py`)
The latest spiking neural network architecture achieves 100% reconstruction accuracy through a combination of rate coding and population coding, successfully converting continuous embeddings into neural spikes. The system implements population coding with temporal position encoding and rate-based spike generation. The system features phase-shifted spiking attention mechanisms and leaky integrate-and-fire neurons, managed through careful membrane potential control systems.

### 10. Advanced Neural Architecture (`cerebro8.py`)
The most sophisticated neural simulation environment implements a multi-layer cortical architecture with integrated working memory systems. The architecture features visualization tools for network analysis and implements Spike-Timing-Dependent Plasticity (STDP) for biologically-inspired learning processes.

## Technical Achievements & Innovations

### Embedding Compression Pipeline
Starting with continuous embeddings from FastText and OpenAI's ada-002, we developed a binary autoencoder using sigma-delta quantization to preserve Euclidean distances. The system progressed to handle concatenated embedding models and ultimately achieved a novel compression using Microsoft's 1.58 bit LLM compression technique, resulting in 50-100x compression savings while maintaining 100% reconstruction accuracy.

### Distributed Computing
We implemented multi-GPU training support with efficient data parallel processing pipelines. The streaming dataset processing and distributed sampler implementation enable scalable training across multiple computing nodes. The distributed processing pipeline incorporates features such as mixed-precision training, automated GPU memory management, and efficient batch processing with error handling. The binary embedding system achieves significant memory reduction while preserving semantic relationships through quantization schemes.

### Biological Neural Architecture Design
Key innovations include the fusion of traditional transformers with spiking neural networks, implementation of novel population coding mechanisms, and development of phase-shifted attention mechanisms. The architecture maintains biological plausibility while optimizing for computational efficiency.

We implemented neural encoding mechanisms using high-density population coding and biologically-plausible firing rates. The system successfully converts complex language embeddings into spike-based representations while maintaining semantic relationships. The integration of evolutionary network architecture with reinforcement learning principles demonstrates a novel approach to training brain-like neural networks.

### Performance Optimization
The binary embedding systems achieve a 50-100x reduction in memory usage when converting from continuous to spiking representations (with 5% sparsity and Microsoft's 1.58 bit LLM compression techniques). The integration of spiking networks significantly reduces computational complexity, while the distributed training approach enables scalability.

## Project Progression Highlights
The research evolved from a basic neuroevolution model with just 3 neurons to increasingly complex implementations. Key milestones include tokenizing an OpenWebText corpus, converting continuous embeddings to binary representations using 8 H100 GPUs, and implementing compression techniques. The project culminated in developing the largest known Liquid State Machine, featuring cortical layers and a midbrain reservoir for biologically realistic organization.

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
midhun.sadanand@yale.edu
