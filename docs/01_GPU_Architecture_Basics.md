# 01. GPU Architecture Basics: Under the Silicon Hood (10/10 Enterprise Depth)

Before optimizing deep learning workloads, we must understand the physical constraints, logic translations, and topologies of the physical host. Treating a GPU as a black-box "speedup device" leads to catastrophic underutilization and severe network bottlenecks on distributed jobs.

---

## 1. The Compute Hierarchy: From Python to Silicon

When you write PyTorch code, how does `torch.matmul(A, B)` actually execute?

1. **Eager Execution / JIT Compilation:** Frameworks compile high-level Python operations into a C++ backend (ATen for PyTorch).
2. **CUDA Kernels:** The ATen backend invokes highly optimized pre-written C++ kernels (found in cuBLAS or cuDNN). 
3. **PTX (Parallel Thread Execution):** The CUDA kernels are compiled by the `nvcc` compiler into PTX. PTX is an intermediate assembly language that NVIDIA uses. PTX guarantees that code compiled today will theoretically work on next-generation GPUs.
4. **SASS (Streaming ASSembly):** The NVIDIA driver JIT-compiles the PTX down to SASS (machine code specific to the exact microarchitecture, e.g., Hopper or Ampere). This is what actually runs on the SMs.

### The Streaming Multiprocessor (SM)
The fundamental unit of a GPU. An H100 GPU contains up to 132 SMs.
*   **Thread Blocks & Grids:** A kernel execution spawns a "Grid" of "Thread Blocks". The GPU scheduler assigns Thread Blocks to the SMs.
*   **The Warp Scheduler:** Threads are bundled into "Warps" of exactly 32 threads. An SM executes instructions warp by warp. *Crucial rule*: If one thread in your warp takes an `if` branch and 31 threads take the `else` branch, the hardware **must execute both branches serially**, masking out inactive threads (Warp Divergence). You lose massive performance.

### Tensor Cores vs CUDA Cores
*   **CUDA Cores:** Standard FMA (Fused-Multiply-Add) units processing 1 number per clock.
*   **Tensor Cores:** Matrix Math ALUs. A Hopper Tensor Core can perform a 4x4 matrix multiply in a single hardware instruction cycle (e.g., $D = A \times B + C$). 
    *   *Operational Truth:* If your `nvidia-smi` GPU-Util is 100%, but your `Tensor Core Util` (viewable via `nsys` or `dcgm`) is 0%, your deep learning job is executing on CUDA cores and running 10x to 30x slower than it should.

---

## 2. Memory Topologies & Bandwidth Mathematics

The vast majority of Deep Learning operations (especially Inference) are **Memory-Bound**, not **Compute-Bound**. The concept is simple: The SM can do math faster than it can fetch data from VRAM.

### The Memory Math Formula
You can calculate the true maximum bandwith of your GPU using hardware spec sheets:
$$\text{Memory Bandwidth (GB/s)} = \left(\frac{\text{Memory Interface Width (bits)}}{8}\right) \times \text{Memory Clock (Hz)} \times \text{Data Rate (DDR)}$$

**Example (A100 80GB configuration):**
*   Memory Width = 5120 bits
*   Clock = 1593 MHz
*   $ \frac{5120}{8} \times 1593 \times 2 = \text{Approx 2039 GB/s} $

If your PyTorch profiling shows you are reading/writing at 1.8 TB/s, you are hitting the "Memory Wall"—the physical limit of the HBM chips. No amount of software tuning will make the matrix load faster unless you fuse operators (e.g., FlashAttention).

---

## 3. NVLink, NVSwitch, and NUMA Topologies

When building 8-GPU nodes (like HGX chassis), the physical wiring of the PCIe lanes and NVLinks dictate cluster performance. 

### Checking Topologies with `nvidia-smi topo -m`
If your model requires `DistributedDataParallel`, the GPUs must share gradients. If they sync over a PCIe bridge instead of an NVLink, your epoch times will 10x.

Run this on any high-end AI node:
```bash
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    NV12    NV12    0-15            0
GPU1    NV12     X      NV12    NV12    0-15            0
GPU2    NV12    NV12     X      NV12    16-31           1
GPU3    NV12    NV12    NV12     X      16-31           1
```
**How to read this:**
*   `NV12`: GPUs communicate via 12 NVLinks (approx 600 GB/s on A100). This is optimal.
*   `PIX` / `PXB`: GPUs communicate via the PCIe bus underneath a CPU bridge (approx 32 GB/s to 64 GB/s). This is catastrophic for tightly coupled distributed training.

### The NUMA Architecture Trap
Look at the `NUMA Affinity` column above. 
A dual-CPU node has Non-Uniform Memory Access (NUMA). GPU0 and GPU1 are wired to CPU0. GPU2 and GPU3 are wired to CPU1.

*   If GPU0 needs to read system RAM (Host-to-Device transfer) that currently lives on the CPU1 memory bank, it has to traverse the QPI/UPI link connecting the two CPUs. 
*   **The Fix:** Always pin your dataloader processes to the CPU socket that directly owns the GPU. (e.g., using PyTorch's `CUDA_VISIBLE_DEVICES` alongside `taskset -c 0-15` or Kubernetes CPU Manager Policies). Failure to align NUMA topologies results in silent I/O starvation and dropped GPU utilization spikes.
