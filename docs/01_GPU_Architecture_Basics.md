# 01. GPU Architecture Basics: Under the Silicon Hood

Before optimizing deep learning workloads, we must understand the physical constraints and capabilities of the hardware. Treating a GPU as a black-box "speedup device" leads to catastrophic underutilization.

## CPU vs. GPU: The Core Difference

*   **CPU (Central Processing Unit):** Designed for low latency. Think of it as a few extremely fast, sophisticated PhDs who can solve complex sequential math problems very quickly.
*   **GPU (Graphics Processing Unit):** Designed for high throughput. Think of it as thousands of high-schoolers who can only do basic arithmetic, but they can all do it simultaneously.

Deep Learning is fundamentally matrix multiplication (basic arithmetic), making it perfect for the GPU's highly parallel architecture.

## Inside an NVIDIA GPU

Modern NVIDIA GPUs (Ampere, Hopper architectures) are composed of several key components:

### 1. Streaming Multiprocessors (SMs)
The SM is the heart of the GPU. A single GPU contains many SMs (e.g., an A100 has 108 SMs). Each SM manages hundreds of threads concurrently and executes instructions in parallel. 
*   **Warp:** Threads are grouped into "warps" of 32 threads. The SM executes one instruction across the entire warp simultaneously (SIMT: Single Instruction, Multiple Threads). **If threads in a warp diverge structurally (e.g., complex `if/else` statements), performance tanks.**

### 2. Computing Cores
Within the SM, different specialized execution units exist:
*   **CUDA Cores:** The standard FP32/INT32 execution units. Good for general-purpose parallel math.
*   **Tensor Cores:** Specialized execution units designed **exclusively for matrix multiplication**. They can perform massive 4x4 or 8x8 matrix multiplies in a single clock cycle. *If your AI workload is not using Tensor Cores, you are wasting 80% of your GPU's potential.*

### 3. Memory Hierarchy (The Real Bottleneck)
In AI workloads, the bottleneck is rarely compute (TFLOPS); it is almost always **memory bandwidth**. You can't compute if you can't feed data to the cores fast enough.
*   **L1 Cache / Shared Memory:** Lives inside the SM. Very small, exceptionally fast.
*   **L2 Cache:** Shared across all SMs. 
*   **HBM (High Bandwidth Memory):** Replaces traditional GDDR memory. HBM is physically stacked directly next to the GPU chip to provide massive bandwidth (e.g., H100 provides over 3 TB/s). This is your "vRAM". 

## Interconnects: Scaling Beyond a Single GPU

When your LLM doesn't fit in 80GB of vRAM, you need multiple GPUs. How they talk to each other is critical.

*   **PCIe (Peripheral Component Interconnect Express):** Connects the GPU to the CPU and the motherboard. Maxes out around 64 GB/s (PCIe Gen5) per lane configuration. It is too slow for heavy GPU-to-GPU sync.
*   **NVLink:** NVIDIA's proprietary GPU-to-GPU bridge. It allows GPUs to bypass the CPU entirely, providing speeds up to 900 GB/s (A100) or 1.8 TB/s (H100).
*   **NVSwitch:** Forms a topology allowing all GPUs in a single physical node to communicate at full NVLink speeds simultaneously.
*   **RDMA / RoCE / InfiniBand:** When scaling across *multiple nodes* (e.g., two physical servers), GPUs use RDMA (Remote Direct Memory Access) over InfiniBand networks to talk over the wire without CPU interruption.

---
**Author's Note:** *As an AI operations engineer, one of the most common mistakes I see is provisioning a 8x GPU node without NVLink. Distributed training like `DistributedDataParallel`(DDP) will fall back to PCIe and your expensive GPUs will spend 90% of their time waiting on the network. Always verify your topology with `nvidia-smi topo -m`.*
