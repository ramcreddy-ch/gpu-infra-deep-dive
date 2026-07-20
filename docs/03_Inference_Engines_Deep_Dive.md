# vLLM Internals, TensorRT-LLM Internals, SGLang Internals, llama.cpp Internals & Inference Optimization

> *Inference engines are the runtime layer between your model weights and your API endpoint. Choosing the right engine — and understanding its internals — determines your throughput, latency, cost, and operational complexity. This chapter dissects the four most important inference engines from architecture to kernel level.*

---

## 1. What Problem Does This Solve?

### Why Can't You Just Use PyTorch for Inference?

```python
# Naive PyTorch inference
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")
output = model.generate(input_ids, max_new_tokens=512)
```

This works for prototyping but is **catastrophically inefficient for production**:

1. **No KV Cache management:** PyTorch recomputes attention for all previous tokens on every step
2. **No batching:** Processes one request at a time
3. **No memory optimization:** Allocates max sequence length upfront, wastes memory
4. **No kernel fusion:** Runs hundreds of small CUDA kernels with launch overhead between each
5. **No continuous batching:** Can't add new requests mid-generation
6. **No speculative decoding:** No acceleration for predictable sequences

**Real numbers:** PyTorch native generates ~5 tokens/second for Llama-70B on an H100. vLLM generates ~2,000 tokens/second on the same hardware. That's a **400x throughput difference**.

---

## 2. vLLM — Architecture & Internals

### What Is vLLM?

vLLM (Virtual LLM) is an open-source inference engine created at UC Berkeley. Its core innovation is **PagedAttention** — managing KV cache memory like an operating system manages virtual memory with page tables.

### vLLM Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        vLLM Server                             │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  API Layer (OpenAI-compatible HTTP server)               │ │
│  │  /v1/chat/completions  /v1/completions  /v1/embeddings  │ │
│  └─────────────────────────┬────────────────────────────────┘ │
│                             │                                  │
│  ┌──────────────────────────▼───────────────────────────────┐ │
│  │  AsyncLLMEngine (main event loop)                        │ │
│  │  ┌────────────────┐  ┌────────────────────────────────┐ │ │
│  │  │  Tokenizer      │  │  Scheduler (per-step)          │ │ │
│  │  │  (HF Tokenizer  │  │  • Maintains WAITING queue     │ │ │
│  │  │   or SentencePc) │  │  • Maintains RUNNING set      │ │ │
│  │  └────────────────┘  │  • Maintains SWAPPED set       │ │ │
│  │                       │  • Decides which seqs to run   │ │ │
│  │                       │  • Preempts if OOM             │ │ │
│  │                       └────────────────────────────────┘ │ │
│  └──────────────────────────┬───────────────────────────────┘ │
│                              │                                 │
│  ┌───────────────────────────▼──────────────────────────────┐ │
│  │  Model Executor (GPU worker processes)                   │ │
│  │                                                          │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │  Block Manager (PagedAttention memory management)  │ │ │
│  │  │                                                    │ │ │
│  │  │  GPU Block Table:                                  │ │ │
│  │  │  ┌──────┬──────┬──────┬──────┬──────┬───────┐    │ │ │
│  │  │  │Blk 0 │Blk 1 │Blk 2 │Blk 3 │ ... │Blk N  │    │ │ │
│  │  │  │16 tok│16 tok│16 tok│16 tok│     │16 tok │    │ │ │
│  │  │  └──────┴──────┴──────┴──────┴──────┴───────┘    │ │ │
│  │  │  CPU Swap Space (for preempted sequences)         │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  │                                                          │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │  Model (Transformer layers on GPU)                 │ │ │
│  │  │  • Attention layers with PagedAttention kernel     │ │ │
│  │  │  • MLP layers with fused SiLU+Multiply            │ │ │
│  │  │  • RMSNorm fused kernels                          │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### PagedAttention — The Core Innovation

**The Problem It Solves:**

In traditional inference, the KV cache for each sequence is allocated as a **contiguous block** of GPU memory. For a 70B model with 80 layers, each token's KV cache entry takes:

```
KV cache per token per layer:
  K: hidden_dim × sizeof(dtype) = 8192 × 2 = 16,384 bytes
  V: hidden_dim × sizeof(dtype) = 8192 × 2 = 16,384 bytes
  Total per layer: 32,768 bytes

Total per token (all layers): 32,768 × 80 = 2,621,440 bytes = 2.5 MB

For max_seq_len = 4096:
  Pre-allocated per sequence: 2.5 MB × 4096 = 10.24 GB ← MASSIVE
```

If you pre-allocate 10 GB per sequence and only generate 100 tokens, you waste 97.6% of the memory.

**How PagedAttention Fixes This:**

Instead of contiguous allocation, PagedAttention divides the KV cache into fixed-size **blocks** (typically 16 tokens each). Blocks are allocated on-demand as the sequence grows, and freed when the sequence finishes.

```
Traditional (contiguous):
  Seq 0: [████████████████████████████████████░░░░░░░░░░░░░░░░░░░]
  Seq 1: [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  Seq 2: [████████████████████████████████████████████████░░░░░░░░]
  FREE:  [░░░░░░░░░░░░░░░░░░░░░░░]  ← Fragmented, unusable
  
  █ = used    ░ = wasted (allocated but unused)

PagedAttention (non-contiguous blocks):
  Physical GPU blocks: [B0][B1][B2][B3][B4][B5][B6][B7][B8][B9]...
  
  Seq 0 page table: B0 → B3 → B7    (3 blocks used)
  Seq 1 page table: B1 → B5          (2 blocks used)
  Seq 2 page table: B2 → B4 → B6 → B8 (4 blocks used)
  Free list: [B9, B10, B11, ...]      ← No fragmentation!
  
  Memory utilization: ~96-99% (vs ~40-60% with contiguous)
```

### vLLM Scheduling Algorithm (Per-Step)

```python
# Simplified vLLM scheduler logic (runs before every decode step)
def schedule(self):
    # Phase 1: Can any RUNNING sequences continue?
    # Check if there are free KV cache blocks for the next token
    running_scheduled = []
    for seq in self.running:
        if self.block_manager.can_append_slot(seq):
            running_scheduled.append(seq)
        else:
            # Not enough blocks → PREEMPT (swap to CPU or recompute)
            self.preempt(seq)
    
    # Phase 2: Can any WAITING sequences start?
    # These are new requests that haven't started yet
    waiting_scheduled = []
    for seq in self.waiting:
        if self.block_manager.can_allocate(seq):
            self.block_manager.allocate(seq)
            waiting_scheduled.append(seq)
        else:
            break  # No more memory for new sequences
    
    # Phase 3: Can any SWAPPED sequences resume?
    swapped_scheduled = []
    for seq in self.swapped:
        if self.block_manager.can_swap_in(seq):
            self.block_manager.swap_in(seq)
            swapped_scheduled.append(seq)
    
    return running_scheduled + waiting_scheduled + swapped_scheduled
```

### vLLM Production Deployment

```bash
# Production vLLM launch command
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-requests \
    --port 8000
```

**Key parameters explained:**
- `--tensor-parallel-size 4`: Shard model across 4 GPUs (NVLink required for performance)
- `--gpu-memory-utilization 0.90`: Reserve 90% of VRAM for model + KV cache
- `--max-num-seqs 256`: Maximum concurrent sequences in the batch
- `--enable-chunked-prefill`: Process long prefills in chunks, interleaved with decodes
- `--enable-prefix-caching`: Cache common prompt prefixes (system prompts)

---

## 3. TensorRT-LLM — NVIDIA's Optimized Engine

### Architecture

TensorRT-LLM compiles the model into an optimized TensorRT engine at build time. It is not an interpreter like vLLM — it generates custom CUDA kernels specifically for your model and hardware.

```
Build Phase (offline, run once):
  ┌───────────────┐     ┌────────────────┐     ┌──────────────┐
  │ HuggingFace   │ ──→ │ TRT-LLM Python │ ──→ │ TensorRT     │
  │ Model Weights │     │ Model Builder  │     │ Engine File  │
  │ (safetensors) │     │ (define graph) │     │ (.engine)    │
  └───────────────┘     └────────────────┘     └──────────────┘
                                                      │
                                                      ▼
Runtime Phase (production):                    ┌──────────────┐
  ┌─────────────┐     ┌──────────────────┐    │ Optimized    │
  │ API Request │ ──→ │ Triton Inference  │ ─→│ CUDA Kernels │
  │             │     │ Server + TRT-LLM │    │ (fused ops)  │
  │             │ ←── │ Backend          │ ←──│              │
  └─────────────┘     └──────────────────┘    └──────────────┘
```

### Why TensorRT-LLM Is Faster

1. **Kernel fusion:** Combines multiple operations (LayerNorm + Linear + Activation) into single kernels
2. **Custom GEMM kernels:** Uses NVIDIA's hand-tuned matrix multiply kernels for specific shapes
3. **FP8 quantization:** Native support for H100 FP8 Tensor Cores (2x throughput over FP16)
4. **In-flight batching:** NVIDIA's version of continuous batching
5. **Paged KV cache:** Same concept as vLLM's PagedAttention

### Build and Deploy

```bash
# Build TRT-LLM engine for Llama-3.1-70B
python convert_checkpoint.py \
    --model_dir /models/Llama-3.1-70B \
    --output_dir /engines/llama-70b-ckpt \
    --dtype float16 \
    --tp_size 4

trtllm-build \
    --checkpoint_dir /engines/llama-70b-ckpt \
    --output_dir /engines/llama-70b-trt \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_seq_len 8192 \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --use_fused_mlp enable
```

### TensorRT-LLM vs vLLM Comparison

| Feature | vLLM | TensorRT-LLM |
|---|---|---|
| **Ease of use** | `pip install vllm` | Complex build pipeline |
| **Model support** | 100+ models | ~30 models |
| **Performance** | Very good | Best on NVIDIA (10-30% faster) |
| **FP8 support** | Yes (via FBGEMM) | Native (best implementation) |
| **Hardware lock-in** | AMD ROCm support | NVIDIA only |
| **Serving** | Built-in OpenAI server | Requires Triton or custom server |
| **Community** | Very active OSS | NVIDIA-driven |
| **Best for** | Flexibility, rapid iteration | Maximum throughput on NVIDIA |

---

## 4. SGLang — The Rising Competitor

SGLang (Structured Generation Language) optimizes for **structured output generation** — JSON, code, constrained decoding. Its innovations:

1. **RadixAttention:** Extends prefix caching to a radix tree data structure. Multi-turn conversations share KV cache entries across different conversation branches.
2. **Constrained Decoding Optimization:** When generating JSON with a known schema, SGLang prunes the vocabulary at each step, reducing the softmax computation.
3. **Overlap Scheduling:** Overlaps CPU scheduling with GPU execution more aggressively than vLLM.

```bash
# SGLang deployment
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --port 8000
```

---

## 5. llama.cpp — CPU & Edge Inference

llama.cpp is written in pure C/C++ with optional CUDA, Metal (Apple), and Vulkan backends. It is designed for running LLMs on consumer hardware.

### Why llama.cpp Matters

- Runs on **CPU** (AVX2/AVX-512), **Apple M-series** (Metal), **AMD GPUs** (Vulkan)
- GGUF quantization format allows 2-bit to 8-bit quantization
- Can run Llama-3.1-70B on a MacBook Pro M3 Max (96GB RAM) at ~10 tokens/sec
- Used by: Ollama, LM Studio, GPT4All, Jan.ai

### GGUF Quantization Types

```
Q2_K:   2-bit, ~60% quality retention, smallest size
Q3_K_M: 3-bit medium, ~75% quality
Q4_0:   4-bit, round-to-nearest, ~85% quality
Q4_K_M: 4-bit K-quant medium, ~90% quality  ← Most popular
Q5_K_M: 5-bit K-quant medium, ~95% quality  ← Best balanced
Q6_K:   6-bit, ~98% quality
Q8_0:   8-bit, ~99.5% quality, largest
F16:    Full FP16, baseline
```

---

## 6. Production Incident Scenarios

### Incident: "vLLM OOMKilled After Running Fine for 2 Hours"
**Symptoms:** Pod restarts with OOMKilled. `nvidia-smi` showed 99% VRAM used right before crash.
**Root Cause:** Prefix caching was enabled and the cache grew unbounded. Long-lived system prompts accumulated in the prefix cache, leaving insufficient memory for new request KV caches.
**Fix:** Set `--max-num-seqs` lower to limit concurrent requests, or increase `gpu-memory-utilization` headroom. In newer vLLM versions, the prefix cache has automatic eviction.

### Incident: "TensorRT-LLM Engine Crashes on Long Inputs"
**Symptoms:** `RuntimeError: CUDA error: out of memory` on inputs > 4000 tokens.
**Root Cause:** The TRT engine was built with `--max_input_len 4096` but the actual input was 4,100 tokens. TensorRT allocates memory at build time based on these maxes.
**Fix:** Rebuild with higher `--max_input_len`. TensorRT-LLM is NOT flexible at runtime — you must rebuild for different shapes.

### Incident: "vLLM Throughput Drops 5x Under Heavy Load"
**Symptoms:** At 50 concurrent users, throughput is 2,000 tok/s. At 500 concurrent users, throughput drops to 400 tok/s.
**Root Cause:** Too many concurrent sequences cause KV cache preemption (swap to CPU). The preemption overhead dominates.
**Fix:** Implement a load balancer with a queue. Limit vLLM's `max-num-seqs` to 128, and use an external queue (Redis/SQS) to buffer excess requests.

---

## Summary

```
Engine Selection Guide:
  Need maximum throughput on NVIDIA?     → TensorRT-LLM
  Need flexibility + good performance?   → vLLM
  Need structured output / multi-turn?   → SGLang
  Need CPU/Mac/edge deployment?          → llama.cpp (GGUF)
  Need managed service?                  → AWS Bedrock, Azure OpenAI, GCP Vertex

Key vLLM Parameters:
  --tensor-parallel-size     # GPUs per model instance
  --gpu-memory-utilization   # % VRAM for model + KV cache (0.85-0.95)
  --max-num-seqs             # Max concurrent requests (tune to avoid preemption)
  --enable-prefix-caching    # Cache system prompts across requests
  --enable-chunked-prefill   # Interleave prefill and decode for lower TTFT

Throughput Hierarchy (same hardware, same model):
  TensorRT-LLM FP8 > TensorRT-LLM FP16 > SGLang ≈ vLLM > PyTorch native
  Typical: 2,500     2,000                1,800    1,700   5 tok/s
```

---

*Next: [04 — KV Cache, Paged Attention, Flash Attention, Speculative Decoding →](04_KV_Cache_Attention_Deep_Dive.md)*
