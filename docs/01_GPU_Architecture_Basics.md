# GPU Architecture, CUDA Cores, Tensor Cores, Streaming Multiprocessors, VRAM, HBM, CUDA Programming Model & GPU Memory Hierarchy

> *This is the foundational chapter. Every single concept in AI infrastructure — from why vLLM uses PagedAttention to why NCCL hangs happen — traces back to understanding GPU architecture at the silicon level. If you don't understand SMs, warps, and the memory hierarchy, you are debugging AI systems blind.*

---

## 1. What Problem Does This Solve?

### Why GPUs Exist for AI

CPUs were designed for **latency-sensitive, sequential workloads**. A modern Intel Xeon has 32-64 cores, each incredibly powerful, with deep branch prediction, out-of-order execution, and massive caches. A CPU core can execute any arbitrary instruction sequence efficiently.

GPUs were designed for **throughput-sensitive, parallel workloads**. An NVIDIA H100 has **16,896 CUDA cores** and **528 Tensor Cores**. Each individual core is weak compared to a CPU core — no branch prediction, no out-of-order execution, tiny caches. But together, they can perform **thousands of matrix multiplications simultaneously**.

### Why This Matters for AI

Neural networks are fundamentally **matrix multiplication machines**. A forward pass through a transformer layer is:

```
Attention:  Q×K^T → Softmax → ×V    (GEMM operations)
FFN:        x×W1 → GELU → ×W2       (GEMM operations)
```

A single GPT-4-class forward pass requires approximately **1.8 trillion floating-point operations**. On a CPU, this takes minutes. On an H100 GPU, it takes **milliseconds**.

### What Problems Existed Before GPUs in AI?

Before GPU computing (pre-2012):
- Training ImageNet took **weeks** on CPU clusters
- The 2012 AlexNet breakthrough used 2 NVIDIA GTX 580s and trained in 5 days (vs. months on CPUs)
- Research was bottlenecked by compute, not by ideas
- HPC clusters existed (Cray, IBM) but cost millions and weren't accessible

### Alternatives to NVIDIA GPUs

| Technology | Vendor | Strengths | Weaknesses | Who Uses It |
|---|---|---|---|---|
| **NVIDIA GPU (CUDA)** | NVIDIA | Ecosystem, CUDA, Tensor Cores, NVLink | Expensive, supply constrained | Everyone |
| **AMD MI300X** | AMD | 192GB HBM3, competitive FLOPS | ROCm ecosystem immature | Microsoft, Meta |
| **Google TPU** | Google | Best for large-scale training, custom ISA | Only on GCP, limited flexibility | Google, Anthropic |
| **AWS Trainium/Inferentia** | AWS | 50% cheaper for training/inference | Only on AWS, limited model support | AWS customers |
| **Intel Gaudi 3** | Intel | Integrated Ethernet (no InfiniBand needed) | Ecosystem far behind | Intel ecosystem |
| **Cerebras WSE-3** | Cerebras | Entire wafer as single chip (4 trillion transistors) | Exotic, hard to program | Research labs |
| **Graphcore IPU** | Graphcore | Novel architecture for sparse workloads | Company restructured, uncertain future | Limited |

**Why NVIDIA is preferred:** CUDA has a 16-year ecosystem advantage. PyTorch, TensorFlow, NCCL, TensorRT, Triton, cuDNN — everything is CUDA-first. Switching cost is enormous. AMD's ROCm is improving but still has gaps in operator coverage, debugging tools, and distributed training libraries.

### Real Industry Examples

- **OpenAI** trains GPT models on clusters of 10,000+ NVIDIA H100 GPUs connected via InfiniBand
- **Meta** built the RSC (Research SuperCluster) with 16,000 A100 GPUs for LLaMA training
- **Tesla** built Dojo but still relies heavily on NVIDIA A100/H100 for Autopilot training
- **Google** uses TPU v5p pods (8,960 chips) for Gemini training but also runs NVIDIA GPUs for external workloads

---

## 2. Internal Architecture

### NVIDIA GPU Architecture (H100 SXM as Reference)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA H100 SXM (Hopper Architecture)               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      GPC 0 (Graphics Processing Cluster)            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ... (9 TPC)  │   │
│  │  │  TPC 0   │ │  TPC 1   │ │  TPC 2   │ │  TPC 3   │              │   │
│  │  │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │              │   │
│  │  │ │ SM 0 │ │ │ │ SM 2 │ │ │ │ SM 4 │ │ │ │ SM 6 │ │              │   │
│  │  │ └──────┘ │ │ └──────┘ │ │ └──────┘ │ │ └──────┘ │              │   │
│  │  │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │              │   │
│  │  │ │ SM 1 │ │ │ │ SM 3 │ │ │ │ SM 5 │ │ │ │ SM 7 │ │              │   │
│  │  │ └──────┘ │ │ └──────┘ │ │ └──────┘ │ │ └──────┘ │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │  GPC 1 ... GPC 7  (8 GPCs total, 132 SMs total)         │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        L2 Cache (50 MB)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ HBM3   │ │ HBM3   │ │ HBM3   │ │ HBM3   │ │ HBM3   │ │ HBM3   │       │
│  │ Stack 0 │ │ Stack 1 │ │ Stack 2 │ │ Stack 3 │ │ Stack 4 │ │ Stack 5 │       │
│  │ ~13.3GB│ │ ~13.3GB│ │ ~13.3GB│ │ ~13.3GB│ │ ~13.3GB│ │ ~13.3GB│       │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
│                    Total: 80 GB HBM3 @ 3.35 TB/s                          │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐     │
│  │  NVLink 4.0      │  │  PCIe Gen5 x16   │  │  NVSwitch (external) │     │
│  │  18 links        │  │  128 GB/s         │  │  900 GB/s total      │     │
│  │  900 GB/s total  │  │  (bidirectional)  │  │  GPU-to-GPU          │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Streaming Multiprocessor (SM) — The Fundamental Compute Unit

Every GPU operation ultimately executes on an SM. Understanding the SM is understanding the GPU.

```
┌──────────────────────────────────────────────────────────────┐
│                    Streaming Multiprocessor (SM)              │
│                    (H100 Hopper Architecture)                 │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Warp Scheduler × 4                        │  │
│  │  (Each scheduler dispatches 1 instruction per cycle    │  │
│  │   to its assigned set of execution units)              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │  FP32 CUDA Cores     │  │  FP64 CUDA Cores     │        │
│  │  128 per SM           │  │  64 per SM            │        │
│  │  (Single-precision)   │  │  (Double-precision)   │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │  INT32 Cores          │  │  Tensor Cores (4th Gen)│       │
│  │  64 per SM            │  │  4 per SM              │       │
│  │  (Integer ops)        │  │  (Matrix multiply)     │       │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │  SFU (Special Func)  │  │  Load/Store Units     │        │
│  │  (sin, cos, exp,     │  │  32 per SM            │        │
│  │   rsqrt, log)        │  │  (Memory access)      │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Register File: 256 KB per SM (65,536 × 32-bit)     │  │
│  │  (This is the FASTEST memory — 0 cycle latency)      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Shared Memory / L1 Cache: 228 KB per SM             │  │
│  │  (Configurable split between shared mem and L1)      │  │
│  │  (Shared memory: ~30 cycle latency)                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Tensor Memory Accelerator (TMA)                     │  │
│  │  (New in Hopper: hardware async bulk data copy       │  │
│  │   between global memory and shared memory)           │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Key Architecture Numbers (H100 vs A100 vs A10G vs T4)

| Specification | H100 SXM | A100 SXM | A10G | T4 |
|---|---|---|---|---|
| **Architecture** | Hopper | Ampere | Ampere | Turing |
| **SMs** | 132 | 108 | 72 | 40 |
| **CUDA Cores** | 16,896 | 6,912 | 9,216 | 2,560 |
| **Tensor Cores** | 528 (4th gen) | 432 (3rd gen) | 288 (3rd gen) | 320 (2nd gen) |
| **FP16 Tensor TFLOPS** | 989 | 312 | 125 | 65 |
| **BF16 Tensor TFLOPS** | 989 | 312 | N/A | N/A |
| **FP8 Tensor TFLOPS** | 1,979 | N/A | N/A | N/A |
| **VRAM** | 80 GB HBM3 | 80 GB HBM2e | 24 GB GDDR6X | 16 GB GDDR6 |
| **Memory Bandwidth** | 3,350 GB/s | 2,039 GB/s | 600 GB/s | 300 GB/s |
| **L2 Cache** | 50 MB | 40 MB | 6 MB | 4 MB |
| **NVLink** | 900 GB/s | 600 GB/s | N/A | N/A |
| **TDP** | 700W | 400W | 150W | 70W |
| **PCIe** | Gen5 x16 | Gen4 x16 | Gen4 x16 | Gen3 x16 |

---

## 3. Deep Internal Working

### CUDA Programming Model — Threads, Warps, Blocks, Grids

The CUDA programming model maps directly to hardware. Understanding this mapping is critical for AI infrastructure because it determines how PyTorch kernels execute, why certain batch sizes are optimal, and why GPU utilization metrics can be misleading.

```
CUDA Software Hierarchy          GPU Hardware Mapping
─────────────────────           ──────────────────────
Grid (1 kernel launch)     →    Entire GPU
  └── Block (CTA)          →    1 Streaming Multiprocessor (SM)
        └── Warp (32 threads) →  1 Warp Scheduler execution unit
              └── Thread    →    1 CUDA Core (for that cycle)
```

**The Warp — The Most Important Concept:**

A **warp** is a group of 32 threads that execute in **lockstep** (SIMT — Single Instruction, Multiple Threads). All 32 threads in a warp execute the same instruction at the same time, but on different data.

```
Warp 0 (32 threads):
  Thread 0:  data[0]  × weight[0]
  Thread 1:  data[1]  × weight[1]
  Thread 2:  data[2]  × weight[2]
  ...
  Thread 31: data[31] × weight[31]
  
All 32 threads execute the MULTIPLY instruction simultaneously.
```

**Why this matters for AI:**
- **Warp divergence** (when threads in a warp take different `if/else` branches) kills performance. Both branches execute serially.
- PyTorch tensors are designed to avoid divergence — uniform operations across all elements.
- **Batch size should be a multiple of 32** for optimal warp occupancy.

### GPU Memory Hierarchy — The Most Critical Performance Factor

```
Speed (fast→slow)         Capacity (small→large)        Latency
──────────────────        ──────────────────────         ──────────
Registers (256KB/SM)      Smallest (~256 KB per SM)      ~0 cycles
      ↓
Shared Memory (228KB/SM)  Small (~228 KB per SM)         ~30 cycles
      ↓
L1 Cache (per SM)         (Unified with shared mem)      ~30 cycles
      ↓
L2 Cache (shared)         Medium (50 MB on H100)         ~200 cycles
      ↓
HBM (Global Memory)       Large (80 GB on H100)          ~400-800 cycles
      ↓
PCIe (System RAM)          Huge (Host: 512GB-2TB)        ~10,000+ cycles
```

**Why This Matters for AI Infrastructure:**

The #1 bottleneck in LLM inference is **memory bandwidth**, not compute. Here's why:

For a 70B parameter model (LLaMA-2 70B) in FP16:
- Model weights: 70B × 2 bytes = **140 GB**
- To generate 1 token, every weight must be read from HBM once
- H100 HBM bandwidth: 3,350 GB/s
- Time to read all weights: 140 GB / 3,350 GB/s = **41.8 ms**
- That's only **24 tokens/second** per GPU, even with infinite compute

This is called being **memory-bandwidth bound** — the GPU's Tensor Cores are sitting idle waiting for data from HBM. This is why quantization (INT8, INT4) is so impactful for inference: it reduces the bytes that need to be read.

### HBM (High Bandwidth Memory) — The GPU's Main Memory

HBM is fundamentally different from DDR RAM:

```
DDR5 (CPU):                    HBM3 (GPU):
┌────────┐                     ┌────────┐┌────────┐┌────────┐┌────────┐
│ 1 chip │ on motherboard      │Stack 0 ││Stack 1 ││Stack 2 ││Stack 3 │
│ 64-bit │ bandwidth           │ 8 dies ││ 8 dies ││ 8 dies ││ 8 dies │
│ bus    │                     │stacked ││stacked ││stacked ││stacked │
└────────┘                     │1024-bit││1024-bit││1024-bit││1024-bit│
                               └────────┘└────────┘└────────┘└────────┘
                               On silicon interposer next to GPU die

DDR5: ~50 GB/s per channel     HBM3: ~3,350 GB/s total (6 stacks)
```

HBM achieves massive bandwidth by:
1. **Stacking DRAM dies vertically** (8 dies per stack using TSVs — Through-Silicon Vias)
2. **Wide bus** (1024-bit per stack vs 64-bit for DDR)
3. **Placing memory on the same interposer as the GPU die** (millimeters away, not inches)

### Tensor Cores — The AI Accelerator Within the GPU

CUDA cores perform scalar FP32 operations (1 multiply + 1 add per core per cycle). For a 4×4 matrix multiply, a CUDA core would need 128 operations (4×4×4 multiplies + 4×4×3 adds).

Tensor Cores perform **matrix multiply-accumulate (MMA)** on entire matrices:

```
Tensor Core Operation (4th Gen, Hopper):

D = A × B + C

Where:
  A = 16×16 matrix (FP16/BF16/FP8/INT8)
  B = 16×16 matrix (FP16/BF16/FP8/INT8)
  C = 16×16 matrix (FP32/FP16)
  D = 16×16 matrix (FP32/FP16)

One Tensor Core completes this entire 16×16×16 MMA in ONE cycle.
That's 4,096 FMA operations in a single cycle.
A CUDA core would need 4,096 cycles for the same work.
```

**Tensor Core Precision Support (H100):**

| Data Type | Tensor Core TFLOPS | When to Use |
|---|---|---|
| FP64 | 67 | Scientific computing (not AI) |
| TF32 | 989 | Training (default in PyTorch) |
| BF16 | 989 | Training (recommended) |
| FP16 | 989 | Training/Inference |
| FP8 (E4M3/E5M2) | 1,979 | Inference (2x throughput vs FP16) |
| INT8 | 1,979 | Inference (quantized models) |

**Why BF16 Over FP16 for Training:**

```
FP16: 1 sign bit | 5 exponent bits | 10 mantissa bits
      Range: ±65,504 | Precision: ~3.3 decimal digits

BF16: 1 sign bit | 8 exponent bits | 7 mantissa bits  
      Range: ±3.39×10^38 | Precision: ~2.4 decimal digits

FP32: 1 sign bit | 8 exponent bits | 23 mantissa bits
      Range: ±3.40×10^38 | Precision: ~7.2 decimal digits
```

BF16 has the **same exponent range as FP32** (8 bits), which means gradients don't overflow/underflow during training. FP16's limited range (5 exponent bits) causes loss scaling issues and NaN explosions during training. BF16 trades precision for range, which is the right trade-off for neural network training.

### CUDA Kernel Launch and Execution Flow

When PyTorch runs `output = model(input)`, here's what happens at the hardware level:

```
1. PyTorch Python Layer
   └── torch.matmul(A, B) called
   
2. ATen (C++ Tensor Library)
   └── Dispatches to CUDA backend
   
3. cuBLAS / cuDNN / Custom CUDA Kernel
   └── Selects optimal kernel based on tensor shapes
   └── Example: cublasSgemm for matrix multiply
   
4. CUDA Runtime API
   └── cudaLaunchKernel(kernel_function, grid_dim, block_dim, args, shared_mem, stream)
   └── Queues work on a CUDA Stream
   
5. CUDA Driver API
   └── Submits command buffer to GPU via PCIe MMIO
   
6. GPU Command Processor
   └── Reads command from host
   └── Distributes thread blocks to available SMs
   
7. SM Execution
   └── Warp schedulers pick up warps
   └── Tensor Cores execute MMA operations
   └── Results written to registers → shared memory → global memory (HBM)
   
8. Completion
   └── GPU signals completion via interrupt or polling
   └── PyTorch reads result tensor from GPU memory
```

### PCIe vs NVLink — GPU Interconnects

```
PCIe Gen5 x16:
  ┌──────┐                    ┌──────┐
  │ CPU  │ ←─── 128 GB/s ───→ │ GPU  │
  └──────┘    (bidirectional)  └──────┘
  
  Used for: CPU↔GPU data transfer (loading data, model weights)
  Bottleneck: Copying training data from CPU to GPU

NVLink 4.0 (H100):
  ┌──────┐                    ┌──────┐
  │ GPU0 │ ←─── 900 GB/s ───→ │ GPU1 │
  └──────┘    (bidirectional)  └──────┘
  
  Used for: GPU↔GPU communication (tensor/pipeline parallelism, gradient sync)
  7x faster than PCIe Gen5

NVSwitch (DGX H100):
  All 8 GPUs connected to each other via NVSwitch fabric
  Any GPU can talk to any other GPU at full 900 GB/s
  Total bisection bandwidth: 3.6 TB/s
  
  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │ GPU0 │───│ GPU1 │───│ GPU2 │───│ GPU3 │
  └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
     │          │          │          │
  ┌──┴───────────┴──────────┴──────────┴──┐
  │           NVSwitch Fabric              │
  └──┬───────────┬──────────┬──────────┬──┘
     │          │          │          │
  ┌──┴───┐   ┌──┴───┐   ┌──┴───┐   ┌──┴───┐
  │ GPU4 │───│ GPU5 │───│ GPU6 │───│ GPU7 │
  └──────┘   └──────┘   └──────┘   └──────┘
```

---

## 4. Production Architecture

### Single GPU Inference Server

```
┌─────────────────────────────────────────┐
│  Server: 1x NVIDIA A10G (24GB)         │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  vLLM Server                     │   │
│  │  Model: Llama-3.1-8B (FP16)    │   │
│  │  KV Cache: ~8GB                  │   │
│  │  Model Weights: ~16GB           │   │
│  │  Throughput: ~40 tok/s/user     │   │
│  └─────────────────────────────────┘   │
│                                         │
│  OS: Ubuntu 22.04                       │
│  Driver: NVIDIA 550.x                   │
│  CUDA: 12.4                             │
│  Container: nvcr.io/nvidia/pytorch      │
└─────────────────────────────────────────┘
```

### Multi-GPU Inference (Tensor Parallel)

```
┌──────────────────────────────────────────────────────────┐
│  DGX H100 Node: 8x H100 SXM (640GB total VRAM)         │
│                                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │  GPU 0  │ │  GPU 1  │ │  GPU 2  │ │  GPU 3  │      │
│  │ Layer   │ │ Layer   │ │ Layer   │ │ Layer   │      │
│  │ Shard 0 │ │ Shard 1 │ │ Shard 2 │ │ Shard 3 │      │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘      │
│       └────────────┴──────────┴────────────┘            │
│                  NVSwitch (AllReduce)                     │
│                                                          │
│  Model: Llama-3.1-405B (FP8)                           │
│  Tensor Parallel Degree: 8                               │
│  Each GPU holds: 1/8 of every layer's weight matrices   │
│  Throughput: ~500 tok/s aggregate                        │
└──────────────────────────────────────────────────────────┘
```

### Multi-Node Training Cluster

```
┌────────────────────────────────────────────────────────────────────┐
│                    Training Cluster (256 GPUs)                     │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐       │
│  │   Node 0     │  │   Node 1     │  ...  │   Node 31    │       │
│  │  8x H100     │  │  8x H100     │       │  8x H100     │       │
│  │  NVSwitch    │  │  NVSwitch    │       │  NVSwitch    │       │
│  │  2TB RAM     │  │  2TB RAM     │       │  2TB RAM     │       │
│  └──────┬───────┘  └──────┬───────┘       └──────┬───────┘       │
│         │                 │                       │               │
│  ┌──────┴─────────────────┴───────────────────────┴───────┐      │
│  │              InfiniBand NDR 400 Gbps Network            │      │
│  │              (Full fat-tree topology, non-blocking)     │      │
│  │              SHARP in-network reduction enabled         │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │              Lustre/GPFS Parallel Filesystem              │     │
│  │              100+ GB/s aggregate read throughput          │     │
│  │              Training data: tokenized, pre-shuffled       │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Framework: PyTorch FSDP + FlashAttention 2                       │
│  Data Parallel: 32 nodes                                          │
│  Tensor Parallel: 8 GPUs per node (intra-node NVLink)            │
│  Pipeline Parallel: optional for very large models               │
│  Global Batch Size: 4M tokens                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Kubernetes GPU Cluster Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                    Kubernetes GPU Cluster                             │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Control Plane (3 masters, no GPU)                              │ │
│  │  kube-apiserver │ etcd │ scheduler │ controller-manager         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────┐  ┌─────────────────────────┐           │
│  │  Inference Node Pool    │  │  Training Node Pool      │           │
│  │  5x g5.12xlarge (4xA10G)│  │  4x p5.48xlarge (8xH100)│           │
│  │  Labels:                │  │  Labels:                 │           │
│  │   gpu.nvidia.com/class: │  │   gpu.nvidia.com/class:  │           │
│  │     A10G                │  │     H100                 │           │
│  │   workload: inference   │  │   workload: training     │           │
│  │  Taints:                │  │  Taints:                 │           │
│  │   nvidia.com/gpu=true   │  │   nvidia.com/gpu=true    │           │
│  │   :NoSchedule           │  │   :NoSchedule            │           │
│  └─────────────────────────┘  └─────────────────────────┘           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  System Components                                              │ │
│  │  • NVIDIA GPU Operator (auto: driver, toolkit, device plugin)  │ │
│  │  • DCGM Exporter (GPU metrics → Prometheus)                    │ │
│  │  • Node Feature Discovery (label GPU capabilities)             │ │
│  │  • KEDA (autoscale inference pods based on queue depth)        │ │
│  │  • Prometheus + Grafana (monitoring)                           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 5. Production Use Cases

### How OpenAI Operates GPU Infrastructure
- 10,000+ H100 GPUs in custom clusters built with Microsoft Azure
- Custom InfiniBand fabric (not standard cloud networking)
- Training runs last weeks to months; a single hardware failure can waste millions of dollars
- Implemented custom checkpointing: save model state every 10 minutes to distributed storage
- Batch size tuning: start with small LR and batch, ramp up over the first 2,000 steps

### How Meta Operates GPU Infrastructure
- RSC v2: 16,000 H100 GPUs connected via InfiniBand NDR
- LLaMA-3 405B trained on 16,384 GPUs simultaneously
- **Key insight from Meta's LLaMA paper:** They experienced 419 hardware interruptions during a 54-day training run of LLaMA-3. That's ~8 failures per day. Automated recovery and frequent checkpointing are essential.
- Use FSDP (Fully Sharded Data Parallel) as their primary parallelism strategy

### How NVIDIA Operates GPU Infrastructure
- DGX SuperPOD: 256 DGX H100 nodes (2,048 GPUs) as a single training cluster
- Eos supercomputer: 576 DGX H100 nodes (4,608 GPUs)
- NVLink backbone within node, InfiniBand between nodes
- NVIDIA ships their own ML framework (NeMo) optimized for their hardware

### How Google Operates AI Infrastructure
- TPU v5p pods: 8,960 chips per pod, connected via custom ICI (Inter-Chip Interconnect)
- For GPU workloads (A3 instances with H100), they use GKE with the NVIDIA GPU Operator
- Gemini trained on TPU v4 and v5 pods across multiple data centers

---

## 6. Production Incident Scenarios

### Incident 1: "GPU Utilization is 100% but Training is Slow"
**Symptoms:** `nvidia-smi` shows 100% GPU utilization. Training throughput (samples/sec) is 3x slower than expected.
**Investigation:**
```bash
# Check SM utilization vs GPU utilization
nvidia-smi dmon -s u -d 1
# GPU-Util shows 100% but SM% shows only 15%
```
**Root Cause:** GPU-Util in `nvidia-smi` means "at least one kernel was running during the sampling period." It does NOT mean the GPU is efficiently utilized. Low SM% means the kernels are memory-bandwidth bound — the GPU is spending most of its time waiting for data from HBM.
**Fix:** Enable mixed precision (BF16), increase batch size to improve arithmetic intensity, use FlashAttention.

### Incident 2: "CUDA Out of Memory but nvidia-smi Shows Free VRAM"
**Symptoms:** PyTorch throws `CUDA OOM` but `nvidia-smi` shows 10GB free.
**Root Cause:** PyTorch uses a **caching memory allocator**. It requests large blocks from CUDA and manages them internally. Memory fragmentation means PyTorch can't find a single contiguous block large enough for the allocation, even though total free memory exists.
**Fix:**
```python
# Force PyTorch to release cached memory
torch.cuda.empty_cache()
# Set environment variable to use expandable memory segments (PyTorch 2.0+)
# This dramatically reduces fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### Incident 3: "ECC Error Storm — GPU Producing Wrong Results"
**Symptoms:** Model outputs are garbage. Loss jumps to NaN. No code changes.
```bash
nvidia-smi -q -d ECC
# Shows: Volatile Double Bit ECC Errors: 47
```
**Root Cause:** HBM memory cells are degrading. Single-bit errors are auto-corrected (SECDED ECC), but double-bit errors corrupt data silently. Once DBE count starts climbing, the GPU must be replaced.
**Fix:** Drain the node, mark it unschedulable in Kubernetes, RMA the GPU. Enable DCGM health checks to catch this before production impact.
**Prevention:** Monitor `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` metric. Alert when it goes above 0.

### Incident 4: "Multi-GPU Training Hangs — No Error, No Progress"
**Symptoms:** A distributed training job with 8 GPUs stops making progress. No error messages. No crash. CPU usage drops to near zero. GPUs show 0% utilization.
**Root Cause:** NCCL collective communication (AllReduce) is waiting for all GPUs to synchronize. One GPU finished its forward pass faster than others and is blocking on the AllReduce barrier. The slow GPU has a degraded NVLink.
```bash
# Check NVLink errors
nvidia-smi nvlink --status
# Shows: Link 3: Replay Errors: 145,892 (should be near 0)
```
**Fix:** Drain node, replace NVLink cable or GPU. Set `NCCL_DEBUG=INFO` and `NCCL_IB_TIMEOUT=23` to get better error messages and longer timeouts.

### Incident 5: "Thermal Throttling Disguised as Normal Operation"
**Symptoms:** Training performance slowly degrades over 4 hours. No errors.
```bash
nvidia-smi -q -d TEMPERATURE
# GPU Current Temp: 87°C (Slowdown Threshold: 83°C)
# GPU Shutdown Threshold: 92°C
```
**Root Cause:** The GPU is thermal throttling. It reduces clock speeds to stay within thermal limits. The `nvidia-smi` shows 100% utilization (kernels are always running), but clock speed has dropped from 1980 MHz to 1200 MHz.
**Fix:** Check data center cooling. Ensure server fans are operational. Add spacing between GPU nodes in the rack.
**Prevention:** Alert on `DCGM_FI_DEV_GPU_TEMP` > 80°C and `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` != 0.

### Incident 6: "PCIe Bandwidth Bottleneck — DataLoader Starving GPUs"
**Symptoms:** GPU utilization spikes to 100% for 200ms then drops to 0% for 500ms. Repeating pattern.
**Root Cause:** The data loading pipeline cannot feed data to the GPU fast enough over PCIe. The GPU processes a batch (200ms) and then sits idle waiting for the next batch (500ms).
**Fix:**
```python
# Use multiple DataLoader workers for CPU-side preprocessing
dataloader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True, prefetch_factor=4)
```
`pin_memory=True` allocates page-locked (pinned) memory on the host, enabling async DMA transfers over PCIe (via `cudaMemcpyAsync`). This allows data loading and GPU computation to overlap.

---

## 7. Performance Optimization

### GPU Utilization Optimization
```bash
# Check actual SM occupancy (not the misleading GPU-Util)
dcgmi dmon -e 1009,1010,1011
# 1009: SM Activity
# 1010: SM Occupancy  
# 1011: Tensor Core Active
```

### Memory Optimization Techniques

| Technique | How It Works | Memory Savings | Speed Impact |
|---|---|---|---|
| **Mixed Precision (BF16)** | Store weights in BF16 instead of FP32 | 50% | +30-50% faster |
| **Gradient Checkpointing** | Recompute activations instead of storing | 60-80% | -20-30% slower |
| **FlashAttention** | Fuse attention into single kernel, tile memory | 5-20x less for attention | +2-4x faster |
| **DeepSpeed ZeRO-3** | Shard weights, gradients, optimizer states across GPUs | Divide by #GPUs | -5-10% slower |
| **Quantization (INT8/FP8)** | Reduce weight precision for inference | 50-75% | +50-100% faster |
| **KV Cache Paging (vLLM)** | Page-based KV cache management | Eliminates fragmentation | No speed impact |

### Linux Tuning for GPU Servers

```bash
# 1. Disable GPU address space randomization (prevents CUDA context overhead)
echo 0 > /proc/sys/kernel/randomize_va_space

# 2. Set GPU persistence mode (prevents driver re-init on every CUDA call)
nvidia-smi -pm 1

# 3. Lock GPU clocks at max frequency (prevents dynamic frequency scaling)
nvidia-smi -lgc 1980,1980  # For H100

# 4. Set GPU compute mode to EXCLUSIVE_PROCESS (1 process per GPU)
nvidia-smi -c EXCLUSIVE_PROCESS

# 5. Enable CUDA MPS for multi-process GPU sharing (if needed)
nvidia-cuda-mps-control -d

# 6. HugePages for host memory (reduces TLB misses for large model loads)
echo 4096 > /proc/sys/vm/nr_hugepages
```

---

## 8. Kubernetes Perspective

### GPU Operator Stack

```yaml
# The NVIDIA GPU Operator installs everything needed on K8s:
# 1. NVIDIA Driver (builds as container image)
# 2. NVIDIA Container Toolkit (nvidia-ctk)
# 3. NVIDIA Device Plugin (advertises GPUs to K8s scheduler)
# 4. DCGM Exporter (Prometheus metrics)
# 5. GPU Feature Discovery (labels nodes with GPU info)
# 6. MIG Manager (if using Multi-Instance GPU)

# Install via Helm:
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --set driver.version=550.90.07 \
  --set toolkit.version=1.16.0 \
  --set dcgmExporter.enabled=true \
  --set migManager.enabled=true
```

### Pod GPU Request

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu: 4  # Request 4 GPUs
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm   # Required for NCCL shared memory
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: "16Gi"    # NCCL needs large shared memory
  runtimeClassName: nvidia  # Use NVIDIA container runtime
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### MIG (Multi-Instance GPU) — GPU Sharing

MIG partitions a single physical GPU into up to 7 isolated instances, each with its own memory, L2 cache, and SMs.

```
A100 80GB with MIG enabled:

Option 1: 7x MIG 1g.10gb (7 instances, each 10GB VRAM, 14 SMs)
Option 2: 3x MIG 2g.20gb + 1x MIG 1g.10gb (3 medium + 1 small)
Option 3: 2x MIG 3g.40gb (2 large instances)
Option 4: 1x MIG 7g.80gb (entire GPU, no sharing)

# Create MIG instances
nvidia-smi mig -cgi 9,9,9,9,9,9,9 -C  # 7 x 1g.10gb
```

**When to use MIG vs Time-Slicing:**
- **MIG:** Guaranteed isolation. Use for multi-tenant inference where you need predictable latency.
- **Time-Slicing:** Lower overhead, no memory isolation. Use for development and batch jobs.

---

## 9. Linux Perspective

### NUMA Topology for GPU Servers

```bash
# Check NUMA topology
numactl --hardware
# node 0: CPUs 0-31, 128GB RAM, GPU 0-3
# node 1: CPUs 32-63, 128GB RAM, GPU 4-7

# Bind process to same NUMA node as its GPU
numactl --cpunodebind=0 --membind=0 python train.py --gpu 0,1,2,3
# This prevents cross-NUMA memory access (50% latency penalty)
```

### OOM Killer vs CUDA OOM

These are different events:
- **Linux OOM Killer:** Host RAM is exhausted. The kernel kills the process. `dmesg | grep -i oom`
- **CUDA OOM:** GPU VRAM is exhausted. PyTorch throws an exception. The process may continue.

For GPU workloads, set `vm.overcommit_memory=1` to prevent the OOM killer from killing DataLoader workers that allocate large memory-mapped datasets.

---

## 10. Networking Perspective

### InfiniBand vs Ethernet for GPU Clusters

| Feature | InfiniBand NDR | RoCE v2 (RDMA over Ethernet) |
|---|---|---|
| Bandwidth | 400 Gbps | 400 Gbps (matched) |
| Latency | ~1 μs | ~2-3 μs |
| Congestion Control | Credit-based (lossless) | PFC + ECN (lossy, complex) |
| RDMA | Native | Yes, but requires careful switch config |
| NCCL Support | Excellent | Good (but more tuning needed) |
| Cost | Very expensive ($$$) | Moderate ($$) |
| Who Uses | OpenAI, Meta, NVIDIA | Google, Microsoft, smaller clusters |

### NCCL (NVIDIA Collective Communications Library)

NCCL handles all GPU-to-GPU communication for distributed training. It auto-detects the optimal topology (NVLink, PCIe, InfiniBand) and selects the best algorithm (Ring, Tree, or Direct).

```bash
# Key NCCL environment variables for production
NCCL_DEBUG=WARN                  # INFO is too noisy for production
NCCL_IB_DISABLE=0                # Enable InfiniBand
NCCL_IB_GID_INDEX=3              # For RoCE v2
NCCL_SOCKET_IFNAME=ib0           # Use InfiniBand interface
NCCL_IB_HCA=mlx5                 # Specify HCA device
NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA level
NCCL_P2P_LEVEL=NVL               # Use NVLink for peer-to-peer
NCCL_TOPO_FILE=/etc/nccl/topo.xml # Custom topology file
```

---

## 11-20: Security, Monitoring, Cost, Mistakes, Interview Prep, Hands-on, Comparisons, Best Practices, Roadmap, Summary

*(These sections continue in subsequent chapters. Each service-specific topic — vLLM, Triton, DeepSpeed — will have its own dedicated deep-dive document with the full 20-section treatment.)*

---

## 20. Summary & Cheat Sheets

### Architecture Cheat Sheet

```
GPU Internal Hierarchy:
  Grid → Block → Warp (32 threads) → Thread → CUDA Core / Tensor Core

Memory Hierarchy (fast→slow):
  Registers (0 cycles) → Shared Mem (30) → L2 (200) → HBM (400-800) → PCIe (10,000+)

Key Numbers (H100 SXM):
  132 SMs | 16,896 CUDA Cores | 528 Tensor Cores
  80 GB HBM3 @ 3,350 GB/s | 50 MB L2
  989 TFLOPS BF16 | 1,979 TFLOPS FP8
  900 GB/s NVLink | 128 GB/s PCIe Gen5
  700W TDP

LLM Inference Math:
  Model Size (bytes) = Parameters × bytes_per_param
  Time per token ≥ Model_Size / Memory_Bandwidth
  70B FP16 → 140 GB → 140/3350 = 41.8ms = 24 tok/s (H100)
  70B INT4 → 35 GB → 35/3350 = 10.4ms = 96 tok/s (H100)
```

### nvidia-smi Quick Reference

```bash
nvidia-smi                           # Basic status
nvidia-smi -q -d MEMORY              # Detailed memory info
nvidia-smi -q -d TEMPERATURE         # Thermal status
nvidia-smi -q -d ECC                 # ECC error counts
nvidia-smi dmon -s pucvmet -d 1      # Continuous monitoring
nvidia-smi topo -m                   # GPU topology (NVLink, PCIe)
nvidia-smi nvlink --status           # NVLink health
nvidia-smi mig -lgi                  # List MIG instances
nvidia-smi -pm 1                     # Enable persistence mode
nvidia-smi -lgc 1980,1980            # Lock clock speed
```

### Production Checklist

- [ ] Persistence mode enabled (`nvidia-smi -pm 1`)
- [ ] ECC enabled on all GPUs
- [ ] DCGM Exporter running, Prometheus scraping
- [ ] Alerts on: temperature > 80°C, ECC DBE > 0, XID errors, clock throttling
- [ ] NUMA-aware process binding configured
- [ ] Shared memory (`/dev/shm`) sized appropriately for NCCL
- [ ] HugePages enabled for large model loads
- [ ] GPU driver version pinned and tested before rollout

---

*Next: [02 — Quantization, Batching, Continuous Batching & Dynamic Batching →](02_Quantization_Batching_Deep_Dive.md)*
