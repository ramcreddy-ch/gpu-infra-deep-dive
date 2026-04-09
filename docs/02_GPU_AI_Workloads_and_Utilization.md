# 02. GPU AI Workloads & Utilization

Knowing how a GPU is built is only half the battle. The other half is feeding it efficiently. 

## The Anatomy of vRAM in AI

"CUDA Out of Memory" (OOM) is the most dreaded error in Platform Engineering. Where does the memory go?
When you load a model onto a GPU, vRAM is consumed by:
1.  **Model Weights (Parameters):** The actual parameters of the neural network.
2.  **Optimizer States:** Adam optimizer, for example, typically consumes 2-3x the memory of the model weights because it maintains moving averages and variances.
3.  **Gradients:** Stored during the backward pass.
4.  **Activations:** Intermediate outputs saved during the forward pass so they can be derived in the backward pass. The size of activations scales linearly with Batch Size and Sequence Length.
5.  **CUDA Context / Overhead:** Base allocation simply by initializing CUDA on the device (~1-2GB).

### Precision and Memory Impact
AI doesn't always need decimal-point exactness. Reducing precision reduces vRAM footprint and engages Tensor Cores.
*   **FP32 (Single Precision):** Traditional, takes 4 bytes per parameter. Slow, rarely used for large modern LLMs.
*   **FP16 / BF16 (Half Precision):** Takes 2 bytes per parameter. BF16 (Bfloat16) handles exponent ranges better than FP16, preventing gradient overflow issues. Recommended for modern training.
*   **INT8 / INT4 (Quantization):** Used primarily for Inference. Shrinks memory drastically (1 byte or 0.5 bytes per parameter), but requires careful calibration.

*Rule of Thumb for LLMs:* A 7-Billion parameter model in FP16 will consume ~14GB just for weights. During training with Adam, expect 3-4x that requirement (~50GB+).

## Training vs. Inference Profiles

### 1. Training Profile: High Bandwidth, High Synchronization
Training requires processing massive batches of data repeatedly.
*   **Bottleneck:** Usually NVLink/RDMA bandwidth during gradient synchronization (All-Reduce operations), or compute-bound on massive matrix multiplies. 
*   **Metric to Watch:** `SM Utilization %`. If it is low, you have an I/O bottleneck feeding data from CPU/Disk out to the GPU.

### 2. Inference Profile: Low Latency, KV Cache Heaviness
Inference (especially for LLMs) operates very differently. It generates text token-by-token.
*   **Bottleneck:** Memory Bandwidth (loading weights from HBM to SRAM for every single token) and the **KV Cache**.
*   **The KV Cache Problem:** To prevent recomputing the attention mechanisms for previous tokens, LLMs store past Keys and Values in memory. In long conversations, the KV Cache can consume more memory than the model weights themselves, leading to fragmentation and OOM on inference.
*   **Metric to Watch:** `Memory Bandwidth Utilization` and `KV Cache Size`.

## "Why is my GPU Utilization 100% but performance is slow?"
This is a classic trap. `nvidia-smi` shows GPU-Util at 100%. People assume the GPU is fully maxed out.
**Incorrect.** 
`nvidia-smi` GPU-Util only measures the percentage of time during the past second that *one or more kernels were executing on the GPU*. It does **NOT** measure how many SMs are active.
If only 1 CUDA core is doing `while(true)`, GPU-Util will show 100%.

**The Solution:** Use `DCGM` (Data Center GPU Manager) or `nvtop` to look at **SM Activity** and **Tensor Core Activity**. If Tensor Core activity is 0%, your framework is falling back to standard CUDA cores, resulting in a 10x slowdown.
