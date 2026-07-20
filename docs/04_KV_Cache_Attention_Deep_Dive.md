# KV Cache, Prefix Cache, Paged Attention, Speculative Decoding, Flash Attention & Token Throughput Optimization

> *The Attention mechanism is the heart of the Transformer architecture, but it is also its biggest bottleneck. Understanding how Attention is computed, cached, and optimized is the key to mastering LLM inference throughput.*

---

## 1. What Problem Does This Solve?

### The Attention Bottleneck

In a Transformer, predicting the next token requires looking at all previous tokens.
Self-attention computes the relationship between the current token and every previous token.

```
Attention(Q, K, V) = softmax(QK^T / √d)V
```

To generate the 100th token, the model needs the Keys (K) and Values (V) for tokens 1 through 99.
If we don't cache them, the model must recompute the K and V tensors for all 99 previous tokens just to generate the 100th token. This is an $O(N^2)$ operation and is catastrophically slow.

### The KV Cache Solution

**KV Cache** saves the computed K and V tensors for previous tokens in GPU memory. When generating the 100th token, the model only computes Q, K, and V for the 100th token, and retrieves the previous 99 K and V vectors from the cache.

**The New Problem:** The KV cache consumes massive amounts of GPU VRAM.

```
KV Cache Size = 2 (K and V) × seq_len × num_layers × hidden_size × bytes_per_param
For LLaMA-2 70B (FP16), sequence length 4096:
KV Cache = 2 × 4096 × 80 × 8192 × 2 bytes = 10.7 GB per request!
```

If a single request takes 10.7 GB of VRAM, an 80 GB A100 can only serve 7 concurrent users. Throughput optimization is entirely about managing, shrinking, and bypassing this KV cache overhead.

---

## 2. Internal Architecture

### Attention Computation Data Flow

```
┌────────────────────────────────────────────────────────┐
│               Standard Self-Attention                  │
│                                                        │
│  Input Token (x) ────────────┐                         │
│                              ▼                         │
│                        ┌───────────┐                   │
│                        │ Wq, Wk, Wv│ (Weight Matrices) │
│                        └─────┬─────┘                   │
│          ┌───────────────────┼───────────────────┐     │
│          ▼                   ▼                   ▼     │
│    Query (Q)             Key (K)             Value (V) │
│          │                   │                   │     │
│          │                   ▼                   ▼     │
│          │             ┌───────────┐       ┌───────────│
│          │             │ KV Cache  │       │ KV Cache  │
│          │             │ (append K)│       │ (append V)│
│          │             └─────┬─────┘       └─────┬─────│
│          │                   │                   │     │
│          ▼                   ▼                   │     │
│    ┌───────────────────────────────┐             │     │
│    │     MatMul (Q × K_total^T)    │             │     │
│    └─────────────┬─────────────────┘             │     │
│                  ▼                               │     │
│    ┌───────────────────────────────┐             │     │
│    │     Scale & Softmax           │             │     │
│    └─────────────┬─────────────────┘             │     │
│                  ▼                               ▼     │
│    ┌───────────────────────────────────────────────────│
│    │               MatMul (Scores × V_total)           │
│    └──────────────────────┬────────────────────────────│
│                           ▼                            │
│                      Output Vector                     │
└────────────────────────────────────────────────────────┘
```

### FlashAttention Architecture

FlashAttention optimizes the exact same mathematical operation, but changes the memory access pattern to be **hardware-aware**.

Instead of writing intermediate results (QK^T and Softmax) to the slow HBM (Global Memory), FlashAttention computes the entire attention block in SRAM (Shared Memory) using **Tiling** and **Recomputation**.

```
┌────────────────────────────────────────────────────────┐
│                   FlashAttention                       │
│                                                        │
│  HBM (80 GB, Slow)         SRAM (256 KB, Fast)         │
│  ┌───────────┐             ┌─────────────────────┐     │
│  │ Q, K, V   │─── Tile 1 ─>│ 1. Load Q,K,V chunk │     │
│  │ Matrices  │             │ 2. Q×K^T            │     │
│  └───────────┘             │ 3. Online Softmax   │     │
│      ▲                     │ 4. Score × V chunk  │     │
│      │                     │ 5. Update running   │     │
│      │                     │    output           │     │
│      │                     └────────┬────────────┘     │
│      │                              │                  │
│      └───── Write Final Output ─────┘                  │
│             (NO intermediate writes!)                  │
└────────────────────────────────────────────────────────┘
```

---

## 3. Deep Internal Working

### Why FlashAttention is Fast (Memory IO Math)

Standard Attention Memory IO:
1. Read Q, K from HBM
2. Write S = QK^T to HBM ($O(N^2)$ memory)
3. Read S from HBM
4. Write P = Softmax(S) to HBM ($O(N^2)$ memory)
5. Read P, V from HBM
6. Write O = PV to HBM

**Total HBM Read/Writes:** $O(N^2 + Nd)$

FlashAttention Memory IO:
1. Load blocks of Q, K, V into SRAM
2. Compute S, P, O entirely in SRAM
3. Write final O to HBM

**Total HBM Read/Writes:** $O(Nd)$

By eliminating the $O(N^2)$ HBM read/writes of the attention matrix, FlashAttention provides a **2-4x speedup** and reduces memory usage by 10-20x, enabling context lengths of 128K+.

### Paged Attention Internals (vLLM)

PagedAttention maps logical token blocks to physical GPU memory blocks, completely eliminating external memory fragmentation.

```
Logical Sequence (Tokens)
[0-15] [16-31] [32-47] [48-63]

Page Table (Managed by vLLM Block Manager)
Logical Block 0 -> Physical Block 102
Logical Block 1 -> Physical Block 45
Logical Block 2 -> Physical Block 812
Logical Block 3 -> Physical Block 12

Physical GPU VRAM (Block size = 16 tokens)
[0] [1] ... [12] ... [45] ... [102] ... [812]
```

**During the Attention Kernel Execution:**
The custom PagedAttention CUDA kernel does not expect a contiguous K and V tensor. Instead, it reads the Page Table, fetches the physical blocks sequentially, and computes attention on the fly.

### Prefix Caching Internals

Many requests share the same system prompt (e.g., "You are a helpful assistant...").

1. vLLM hashes the token IDs of the prompt in block-sized chunks (e.g., 16 tokens).
2. It looks up the hash in a global cache table.
3. If a match is found, the new sequence's page table simply points to the *existing* physical block in VRAM.
4. **Reference counting** ensures the block is not deleted until all sequences using it are finished.

**Impact:** A 1000-token system prompt takes 0ms to compute for the 2nd user, and consumes 0 extra VRAM.

### Speculative Decoding Internals

Speculative Decoding breaks the autoregressive bottleneck (generating 1 token at a time) by using a small, fast "Draft Model" to guess the next N tokens, and the large "Target Model" to verify them in parallel.

```
Step 1: Draft Model (e.g., Llama-68M) generates 4 tokens quickly.
Draft: "The" -> "cat" -> "sat" -> "on"

Step 2: Target Model (e.g., Llama-70B) evaluates all 4 tokens in ONE forward pass (parallel).
Target verifies:
P("cat" | "The") > threshold? Yes.
P("sat" | "The cat") > threshold? Yes.
P("on" | "The cat sat") > threshold? Yes.

Result: We generated 4 tokens in the time it usually takes the 70B model to generate 1.
If the Target Model rejects "on", it discards it and provides the correct token, resuming from there.
```

---

## 4. Production Architecture

### High-Throughput Inference Cluster (vLLM + PagedAttention)

```
┌─────────────────────────────────────────────────────────────┐
│                 vLLM Inference Cluster                      │
│                                                             │
│  ┌─────────────────┐    ┌────────────────────────────────┐  │
│  │ Load Balancer   │───>│ API Gateway (Prompt Routing)   │  │
│  │ (Round Robin)   │    │ Hash prefix, route to replica  │  │
│  └─────────────────┘    │ with warm prefix cache         │  │
│                         └─────┬──────────────────────────┘  │
│                               │                             │
│  ┌────────────────────────────▼──────────────────────────┐  │
│  │ Worker Node (8x H100)                                 │  │
│  │                                                       │  │
│  │ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ │  │
│  │ │ Engine 1      │ │ Engine 2      │ │ Engine 3      │ │  │
│  │ │ (TP=2 GPUs)   │ │ (TP=2 GPUs)   │ │ (TP=2 GPUs)   │ │  │
│  │ └───────────────┘ └───────────────┘ └───────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key Architectural Decision: Prefix-Aware Routing**
If you have multiple vLLM replicas, use a gateway (like SGlang's router or a custom Nginx Lua script) to route requests with the same system prompt to the same vLLM replica to maximize Prefix Cache hit rates.

---

## 5. Production Use Cases

- **OpenAI:** Uses a highly optimized proprietary version of PagedAttention and massive KV cache offloading to serve ChatGPT to 100M+ users.
- **Anthropic:** Claude 3's 200K context window is only possible due to RingAttention (a variant of FlashAttention that distributes the sequence across multiple GPUs).
- **Databricks (MosaicML):** Heavily relies on FlashAttention-2 in their training stack to reduce HBM bottlenecks during pre-training.

---

## 6. Production Incident Scenarios

### Incident 1: "Latency Spikes to 10 Seconds Periodically"
**Symptoms:** P99 TTFT (Time To First Token) spikes massively. VRAM usage is at 98%.
**Root Cause:** KV Cache thrashing. vLLM ran out of physical KV cache blocks. It preempted running sequences by swapping their KV caches from GPU HBM to CPU System RAM over PCIe. When the sequences resumed, they had to be swapped back in.
**Fix:** Decrease `gpu_memory_utilization` or decrease `max_num_seqs` to prevent over-subscription of the KV cache.

### Incident 2: "Speculative Decoding Degrades Performance"
**Symptoms:** After enabling Speculative Decoding with a draft model, throughput dropped by 20%.
**Root Cause:** The draft model was not aligned with the target model (e.g., using a base model to draft for an instruct model). The acceptance rate of draft tokens fell below 30%. The overhead of running the draft model outweighed the benefits of parallel verification.
**Fix:** Fine-tune the draft model on the target model's outputs (distillation) to increase the acceptance rate to > 70%, or disable speculative decoding.

### Incident 3: "OOM on 128K Context Despite FlashAttention"
**Symptoms:** Model runs fine up to 64K context, but OOMs at 100K context.
**Root Cause:** FlashAttention eliminates the $O(N^2)$ memory for the attention *matrix*, but the KV Cache itself still grows linearly $O(N)$ with context length. A 100K context KV cache for a 70B model requires ~25 GB of VRAM per request.
**Fix:** Enable **KV Cache Quantization** (e.g., FP8 KV cache) in vLLM to cut the KV cache size in half.

---

## 7. Performance Optimization

### Token Throughput Optimization Checklist

1. **Use FlashAttention-2/3:** It should be enabled by default in modern frameworks. Verify with PyTorch profiler.
2. **Enable PagedAttention:** Use an engine like vLLM or TensorRT-LLM.
3. **Enable Chunked Prefill:** Interleave prefill computations with decode computations to prevent long prompts from pausing ongoing generations.
4. **Quantize the KV Cache:** Use FP8 or INT8 KV cache. This doubles your maximum batch size.
5. **Prefix Caching:** Enable if you have long, shared system prompts.
6. **Tensor Parallelism:** If latency is too high, split the model across GPUs. This divides the KV cache size per GPU and increases memory bandwidth.

---

## 8. Kubernetes Perspective

In Kubernetes, managing KV cache requires careful resource limits:

```yaml
# Avoid OOMKilled by ensuring Host RAM is sufficient for KV Cache Swapping
# If vLLM swaps KV cache to CPU, it needs large host RAM.
resources:
  requests:
    memory: "128Gi" # Request large CPU memory for vLLM swap space
    nvidia.com/gpu: "1"
  limits:
    memory: "256Gi"
```

---

## 9. Linux Perspective

### Pinned Memory for KV Cache Swapping

If vLLM needs to swap KV cache blocks to CPU memory, the transfer speed over PCIe is critical.

- vLLM allocates **pinned (page-locked) memory** on the Linux host.
- Pinned memory bypasses the Linux page cache and allows the GPU to use DMA (Direct Memory Access) to copy the cache directly to host RAM.
- Ensure the container has sufficient `IPC_LOCK` capabilities if custom allocators are used.

---

## 10-20. Advanced Topics, Checklists & Interviews

*(Sections omitted for brevity in this markdown, but focus heavily on the mathematical relationship between Batch Size, Sequence Length, and KV Cache Size during interviews).*

### Interview Cheat Sheet

**Q: Explain the difference between FlashAttention and PagedAttention.**
**A:** FlashAttention is a **compute-level** optimization that fuses the attention matrix calculation to avoid writing $O(N^2)$ intermediate states to HBM. PagedAttention is a **memory-management** optimization that stores the KV Cache in non-contiguous blocks to eliminate memory fragmentation. You use both simultaneously.

**Q: Calculate the KV cache size for a 1000-token sequence.**
**A:** Formula: `2 (K+V) × num_layers × num_heads × head_dim × 2 (bytes for FP16) × seq_len`.
For LLaMA-3 8B: `2 × 32 × 32 × 128 × 2 × 1000 = 524,288,000 bytes ≈ 500 MB`.

---
*Next: [05 — Distributed Training (DDP, FSDP, ZeRO, Megatron) →](05_Distributed_Training_Deep_Dive.md)*
