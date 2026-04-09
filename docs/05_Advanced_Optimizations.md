# 05. Advanced Optimizations (10/10 Enterprise Depth)

If your infrastructure is perfect, network is humming, and hardware is healthy, how do you make the code run 4x faster? By using advanced State-of-the-Art ML Engineering optimizations.

---

## 1. Inference Optimization architectures

Serving LLMs to thousands of concurrent users requires highly specialized inference servers. Standard PyTorch `model.generate()` is criminally slow for production traffic.

### The Inference Engine Contenders:
*   **vLLM:** The open-source king right now. Implements `PagedAttention`. Excellent for throughput (batching hundreds of users together).
*   **TGI (Text Generation Inference by HuggingFace):** Highly optimized Rust-based router, uses FlashAttention natively. Great for low-latency setups.
*   **Triton Inference Server (NVIDIA):** Highly complex, deeply optimized C++ server. Capable of serving TensorFlow, PyTorch, ONNX, and TensorRT engines simultaneously. Supports "Dynamic Inflight Batching". 

### Deep Dive: vLLM and PagedAttention
**The Problem:** The KV Cache stores previous token activations. In standard generation, PyTorch allocates a massive contiguous block of memory for the maximum possible sequence length (e.g., 4096 tokens). If the user generates only 10 tokens, 99.7% of that memory sits physically empty but "reserved." 
Because memory is locked, new concurrent users receive "Server Busy - OOM".

**The PagedAttention Solution:** vLLM borrows OS Virtual Memory pages. It partitions the KV cache into small blocks (e.g., blocks of 16 tokens). As generation progresses, blocks are allocated dynamically on-demand from a centralized memory pool.
*   **Result:** Memory waste drops from 60% to <4%. Throughput limits skyrocket by 3x-4x.

---

## 2. Advanced Training: DeepSpeed & ZeRO

When a model (e.g., LLaMA 70B) is larger than a single node's VRAM (8x 80GB = 640GB), standard Distributed Data Parallel (DDP) completely fails. You must shard the model across multiple nodes.

**DeepSpeed by Microsoft** implements the **ZeRO (Zero Redundancy Optimizer)** memory methodology. By default, every GPU holds a full replica of the optimizer state and weights. ZeRO partitions them.

### Real World DeepSpeed ZeRO-3 JSON Config
In ZeRO Stage 3, the model parameters themselves are sliced. GPU-0 holds layers 1-5, GPU-1 holds layers 6-10, etc. When GPU-0 needs layer 6 for the forward pass, it fetches it rapidly via NVLink broadcast.

```json
{
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```
**Pro-Tip:** Enabling `overlap_comm` is critical. It hides the network latency of transferring layers across nodes by fetching layer $N+1$ *while* the GPU is actively computing layer $N$.

---

## 3. FlashAttention: Tearing Down the Memory Wall

Standard attention mathematical complexity is $O(N^2)$ with respect to sequence length ($N$).
The matrices $Q$ (Query), $K$ (Key), and $V$ (Value) are loaded from slow HBM (High Bandwidth Memory) to fast SRAM (Shared Memory inside the SM).

**The Slowdown:** The attention loop constantly evicts and re-loads these matrices back and forth between HBM and SRAM. Memory bandwidth is choked.

**FlashAttention Solution:** It uses algorithmic tiling. It loads blocks of $Q, K, V$ into SRAM, fuses the attention calculation (matrix multiply, scale, mask, softmax) directly in SRAM, and writes the final output back to HBM exactly once.
*   **Result:** A 2x to 4x speedup in execution time entirely by eliminating hardware memory read/writes. Available in PyTorch 2.0+ via `torch.nn.functional.scaled_dot_product_attention()`.
