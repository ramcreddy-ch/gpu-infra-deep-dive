# 05. Advanced Optimizations

Once the infrastructure is stable, the MLOps engineer's job shifts to software/hardware co-design. How can we make the model run faster without buying more hardware?

## 1. FlashAttention
Standard self-attention memory complexity scales quadratically ($O(N^2)$) with sequence length. 
**The Memory Wall:** The GPU spends more time moving attention matrices from HBM to SRAM than computing them.
**The Solution:** FlashAttention is an algorithm that fuses the attention loop, computing it in blocks. It prevents massive read/writes to the HBM, operating entirely in SRAM.
*   **Impact:** Massive speedup on context lengths > 2048 tokens and zero exact-math degradation. Almost all modern inference servers use this by default today.

## 2. vLLM and PagedAttention
Serving LLMs to concurrent users is difficult because of the **KV Cache fragmentation**. In traditional serving, memory is pre-allocated for the maximum possible sequence length. If a user only sends 5 words, 90% of that memory is wasted.
**The Solution:** PagedAttention (implemented by framework vLLM). It applies Operating System virtual memory paging logic to the KV Cache. Memory is allocated dynamically in non-contiguous blocks.
*   **Impact:** Can increase throughput of an inference server by 2x to 4x simply by packing more requests into the GPU simultaneously.

## 3. TensorRT Compilation
PyTorch models execute operations step-by-step (Eager mode) by default. This incurs Python overhead.
**The Solution:** Pre-compile the model using **NVIDIA TensorRT**. TensorRT analyzes the computational graph and applies **Operator Fusion**. If it sees a Linear layer followed by a ReLU activation, it will fuse them into a single GPU kernel, preventing a round-trip data fetch to the HBM.
*   **Use case:** Production Inference only. TensorRT engines are specific to a target GPU architecture (e.g. an engine compiled for A100 won't work on T4).

## 4. ZeRO (Zero Redundancy Optimizer) via DeepSpeed
Training massive LLMs (70B+) exceeds the memory of any single GPU (even 80GB H100s).
**The Solution:** ZeRO splits the optimizer states, gradients, and parameters across the GPUs in the cluster.
*   **ZeRO Stage 1:** Partitions optimizer states.
*   **ZeRO Stage 2:** Partitions gradients.
*   **ZeRO Stage 3:** Partitions model weights.
*   **ZeRO-Offload:** Shunts memory to the CPU RAM or NVMe when the GPU runs out of vRAM (incredibly slow, but allows massive models to train on low-end hardware).
