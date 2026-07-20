# Quantization (FP16, BF16, INT8, FP8, INT4, 4-bit), Batching, Continuous Batching & Dynamic Batching

> *Quantization and batching are the two most impactful inference optimizations. Quantization reduces the bytes you need to read from HBM (attacking the memory-bandwidth bottleneck). Batching amortizes the fixed cost of reading model weights across multiple requests (attacking arithmetic intensity). Together they can deliver 10-50x throughput improvements over naive serving.*

---

## 1. What Problem Does This Solve?

### The Inference Cost Crisis

Running LLMs in production is extraordinarily expensive. Consider Llama-3.1 70B:

```
Model weights (FP16):     140 GB
H100 HBM bandwidth:      3,350 GB/s
Time to read all weights: 140 / 3350 = 41.8 ms per token
Max throughput:           24 tokens/second (SINGLE user)

At $2/hour for H100:
  Cost per 1M output tokens = $2 × (1,000,000 / 24 / 3600) = $23.15
  OpenAI charges $15/1M output tokens for GPT-4o
  → You CANNOT serve 70B FP16 profitably on a single H100
```

**Quantization solves this:** INT4 quantization reduces the model to 35 GB. Now you read 35 GB instead of 140 GB per token — 4x faster throughput, 4x lower cost per token.

**Batching solves this:** Instead of reading 140 GB of weights to generate 1 token for 1 user, you read 140 GB once and generate 1 token for 32 users simultaneously. The cost of reading weights is amortized across the batch.

### Before Quantization and Batching
- Serving a 70B model required 4-8 GPUs in FP16
- Single-user latency was acceptable, but throughput was terrible
- Cost per token was 5-20x higher than needed for commercial viability
- GPU utilization was < 5% for inference (mostly memory-bandwidth idle)

---

## 2. Quantization — Internal Architecture

### Number Representation Deep Dive

```
FP32 (32 bits total):
┌─────┬──────────┬────────────────────────┐
│Sign │ Exponent │      Mantissa          │
│ 1   │  8 bits  │      23 bits           │
└─────┴──────────┴────────────────────────┘
Range: ±3.4 × 10^38    Precision: ~7.2 decimal digits
Size: 4 bytes per parameter

FP16 (16 bits total):
┌─────┬──────────┬──────────┐
│Sign │ Exponent │ Mantissa │
│ 1   │  5 bits  │ 10 bits  │
└─────┴──────────┴──────────┘
Range: ±65,504           Precision: ~3.3 decimal digits
Size: 2 bytes per parameter

BF16 (16 bits total):
┌─────┬──────────┬─────────┐
│Sign │ Exponent │Mantissa │
│ 1   │  8 bits  │ 7 bits  │
└─────┴──────────┴─────────┘
Range: ±3.4 × 10^38    Precision: ~2.4 decimal digits
Size: 2 bytes per parameter

FP8 E4M3 (8 bits total):
┌─────┬─────┬─────┐
│Sign │Exp  │Mant │
│ 1   │4 bit│3 bit│
└─────┴─────┴─────┘
Range: ±448             Precision: ~1 decimal digit
Size: 1 byte per parameter

FP8 E5M2 (8 bits total):
┌─────┬─────┬────┐
│Sign │Exp  │Mant│
│ 1   │5 bit│2 bi│
└─────┴─────┴────┘
Range: ±57,344          Precision: <1 decimal digit
Size: 1 byte per parameter

INT8 (8 bits total):
┌────────────────┐
│  Integer value │
│  -128 to +127  │
└────────────────┘
Size: 1 byte per parameter

INT4 (4 bits total):
┌────────┐
│ -8..+7 │
└────────┘
Size: 0.5 bytes per parameter (packed, 2 values per byte)
```

### How Quantization Works Internally

**Weight-Only Quantization (most common for inference):**

The idea: convert FP16 weights to INT4/INT8 before inference. During computation, dequantize back to FP16 for the matrix multiply, or use integer Tensor Cores directly.

```python
# Simplified INT8 symmetric quantization
import torch

def quantize_symmetric(weight_fp16: torch.Tensor) -> tuple:
    """Convert FP16 weights to INT8 with a scale factor."""
    # Find the absolute maximum value in the weight tensor
    abs_max = weight_fp16.abs().max()
    
    # Compute scale: maps [-abs_max, abs_max] to [-127, 127]
    scale = abs_max / 127.0
    
    # Quantize: divide by scale, round to nearest integer, clamp
    weight_int8 = torch.clamp(torch.round(weight_fp16 / scale), -128, 127).to(torch.int8)
    
    return weight_int8, scale

def dequantize(weight_int8: torch.Tensor, scale: float) -> torch.Tensor:
    """Convert INT8 weights back to FP16 for computation."""
    return weight_int8.float() * scale

# Example
weight_fp16 = torch.randn(4096, 4096, dtype=torch.float16)  # 32 MB
weight_int8, scale = quantize_symmetric(weight_fp16)          # 16 MB + 4 bytes
# 50% memory savings
```

**Group Quantization (INT4 with groups):**

INT4 has only 16 possible values (-8 to +7). To maintain accuracy, the weight tensor is divided into small groups (typically 128 values), and each group has its own scale and zero-point.

```
Original weight row: [0.15, -0.32, 0.87, -0.04, ...]  (4096 values)

Group 0 (values 0-127):
  scale_0 = max(abs(group_0)) / 7
  zero_point_0 = computed per group
  quantized_0 = round((values - zero_point) / scale)  → [-7..7]

Group 1 (values 128-255):
  scale_1 = max(abs(group_1)) / 7
  ...

Storage: 4096 × 4 bits = 2048 bytes  (weights)
       + 32 groups × (2 + 2) bytes   = 128 bytes (scales + zero_points)
Total: ~2,176 bytes vs 8,192 bytes (FP16)  → 3.75x compression
```

### Quantization Methods Comparison

| Method | Bits | Requires Calibration Data? | Quality (Perplexity) | Speed vs FP16 | Who Uses |
|---|---|---|---|---|---|
| **FP16** | 16 | No | Baseline | 1.0x | Training, high-quality inference |
| **BF16** | 16 | No | ≈ FP16 | 1.0x | Training (recommended) |
| **GPTQ** | 4 | Yes (calibration set) | ~0.5% degradation | 2-4x | Offline quantization, TheBloke models |
| **AWQ** | 4 | Yes (calibration set) | < 0.3% degradation | 2-4x | Better than GPTQ for most models |
| **GGUF** | 2-8 | No (round-to-nearest) | Varies | 1.5-3x | llama.cpp (CPU/Apple Silicon) |
| **FP8** | 8 | No (dynamic scaling) | ~0.1% degradation | 1.8-2x | H100 native, TensorRT-LLM |
| **SmoothQuant** | 8 (W8A8) | Yes | <0.5% degradation | 1.5-2x | Both weights and activations |
| **bitsandbytes NF4** | 4 | No | Good for QLoRA | 2-3x | Fine-tuning with QLoRA |

### The Crucial Difference: Weight-Only vs Weight+Activation Quantization

```
Weight-Only Quantization (W4A16):
  Weights: INT4 (stored in memory, dequantized to FP16 before compute)
  Activations: FP16 (full precision)
  Benefit: Reduces memory and memory bandwidth
  Drawback: Compute is still FP16, can't use INT8 Tensor Cores

Weight+Activation Quantization (W8A8 or W4A4):
  Weights: INT8
  Activations: INT8 (must calibrate activation ranges)
  Benefit: Uses INT8 Tensor Cores (1,979 TFLOPS on H100)
  Drawback: Harder to maintain accuracy, needs calibration data
```

---

## 3. Batching — Deep Internal Working

### Why Batching Is Essential

Without batching, LLM inference is **memory-bandwidth bound**:

```
Single request (batch_size=1):
  Read: 140 GB weights from HBM
  Compute: 140B FLOPs (70B params × 2 FLOPs per param)
  Arithmetic Intensity = 140B FLOPs / 140 GB = 1 FLOP/byte
  
  H100 can do: 989 TFLOPS / 3350 GB/s = 295 FLOPs/byte (potential)
  Achieved: 1 FLOP/byte
  GPU utilization: 1/295 = 0.34% ← TERRIBLE

Batched (batch_size=32):
  Read: 140 GB weights (same — read once for all requests)
  Compute: 140B × 32 = 4.48T FLOPs
  Arithmetic Intensity = 4.48T / 140 GB = 32 FLOPs/byte
  GPU utilization: 32/295 = 10.8% ← Better!
  
Batched (batch_size=256):
  Arithmetic Intensity = 256 FLOPs/byte
  GPU utilization: 256/295 = 86.8% ← Near optimal!
```

**The key insight:** Each additional request in the batch is nearly free — you're already reading the weights anyway. The marginal cost of serving one more user in the same batch is just the KV cache memory.

### Static Batching (Naive)

```
Request Queue:  [R1(50 tokens), R2(200 tokens), R3(30 tokens), R4(150 tokens)]

Static Batch (size=4):
┌────────────────────────────────────────────────────────┐
│  Step 1:  R1  R2  R3  R4  (all start together)       │
│  Step 30: R1  R2  R3✓ R4  (R3 finishes, slot wasted) │
│  Step 50: R1✓ R2  PAD R4  (R1 finishes, slot wasted) │
│  Step 150: PAD R2  PAD R4✓ (R4 finishes)              │
│  Step 200: PAD R2✓ PAD PAD (R2 finishes, 3 slots PAD) │
└────────────────────────────────────────────────────────┘

Problem: After R3 finishes at step 30, its GPU slot is WASTED
         for the remaining 170 steps. Average GPU utilization: ~40%
```

### Continuous Batching (The vLLM Innovation)

```
Continuous batching replaces finished requests with new ones IMMEDIATELY:

Step 1:   [R1, R2, R3, R4]    ← All processing
Step 30:  [R1, R2, R3✓→R5, R4] ← R3 done, R5 starts immediately  
Step 50:  [R1✓→R6, R2, R5, R4] ← R1 done, R6 starts immediately
Step 100: [R6, R2, R5✓→R7, R4] ← R5 done, R7 starts
...

GPU utilization: ~95%+ (slots are NEVER wasted)
```

**How it works internally:**
- The scheduler runs at **every decode step** (not every batch)
- It checks: "Did any request finish? Is there a new request in the queue?"
- If yes, it swaps out the finished request's KV cache and loads the new request's prefill
- The engine processes one decode step for ALL active requests simultaneously

### Dynamic Batching (Triton Inference Server)

Dynamic batching is for non-autoregressive models (BERT, ViT, embeddings) where all inputs can be padded to the same length and processed as a single batch.

```
Request queue (embedding requests):
  t=0ms:  R1 arrives (128 tokens)
  t=5ms:  R2 arrives (64 tokens)
  t=8ms:  R3 arrives (96 tokens)
  t=10ms: Batch timeout reached

Dynamic Batcher forms a batch:
  Pad R2 to 128 tokens, R3 to 128 tokens
  Execute batch [R1, R2_padded, R3_padded] simultaneously
  Return results individually
```

**Triton configuration:**
```
dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 10000  # Wait up to 10ms for more requests
}
```

---

## 4. Production Architecture

### Quantized Model Serving Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│              Quantized Inference Pipeline                     │
│                                                              │
│  1. Model Hub (HuggingFace/S3)                              │
│     └── Original FP16 weights (140 GB for 70B)              │
│                                                              │
│  2. Offline Quantization Job (runs once)                    │
│     ├── Load FP16 weights                                   │
│     ├── Run calibration data (1024 samples)                 │
│     ├── Apply AWQ/GPTQ quantization                         │
│     ├── Save INT4 weights (35 GB)                           │
│     └── Push to Model Registry                              │
│                                                              │
│  3. Inference Server (vLLM)                                 │
│     ├── Load INT4 weights → 35 GB VRAM                     │
│     ├── KV Cache: ~40 GB remaining for requests            │
│     ├── Continuous batching with up to 256 concurrent reqs │
│     └── Throughput: ~2,000 tok/s on single H100            │
│                                                              │
│  4. Load Balancer                                           │
│     └── Routes requests to least-loaded replica             │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Production Incident Scenarios

### Incident 1: "Quantized Model Produces Garbage Outputs"
**Symptoms:** After deploying GPTQ INT4 model, outputs are incoherent.
**Root Cause:** The GPTQ quantization used an incompatible calibration dataset. The model was fine-tuned for code generation but quantized using a news article dataset. The activation distributions during calibration didn't match real usage.
**Fix:** Re-quantize using AWQ with a calibration set representative of production traffic.

### Incident 2: "Latency Spikes Every 30 Seconds"
**Symptoms:** P99 latency spikes from 100ms to 2,000ms every 30 seconds in a continuously-batched vLLM deployment.
**Root Cause:** KV cache memory pressure. When the KV cache fills up, vLLM must preempt (swap out) existing requests to make room for new ones. The swap to CPU memory takes 500ms+.
**Fix:** Reduce `gpu_memory_utilization` from 0.95 to 0.85 to leave headroom, or reduce max concurrent requests.

### Incident 3: "Batch Size 1 is Faster Than Batch Size 32"
**Symptoms:** Increasing batch size makes P50 latency worse.
**Root Cause:** The model is tiny (7B INT4 = 3.5 GB). At batch 32, the KV cache memory dominates and causes HBM thrashing. The GPU is now compute-bound, not memory-bound, and each request waits longer.
**Fix:** For small models, optimize for latency (small batch) on cheaper GPUs (A10G). For large models, optimize for throughput (large batch) on powerful GPUs (H100).

---

## 6. Quantization Decision Tree

```
Is latency-critical (< 50ms TTFT)?
├── YES → FP16 / BF16 (no quantization overhead)
│         or FP8 on H100 (native, no accuracy loss)
└── NO → 
    Is the model > 70B parameters?
    ├── YES → AWQ INT4 (best quality/size for large models)
    │         or GPTQ INT4 (slightly lower quality, widely supported)
    └── NO → 
        Is it running on CPU or Apple Silicon?
        ├── YES → GGUF Q4_K_M or Q5_K_M (llama.cpp)
        └── NO →
            Is it for fine-tuning?
            ├── YES → bitsandbytes NF4 (QLoRA)
            └── NO → FP8 if H100, AWQ INT4 otherwise
```

---

## Summary

```
Quantization Cheat Sheet:
  FP32 → FP16:    2x compression, no quality loss, always do this
  FP16 → FP8:     2x compression, <0.1% quality loss, H100 only, native
  FP16 → INT8:    2x compression, <0.5% quality loss, requires calibration
  FP16 → INT4:    4x compression, 0.3-1% quality loss, GPTQ/AWQ
  
Batching Cheat Sheet:
  Static batching:     Pad to longest sequence, waste compute on padding
  Dynamic batching:    Wait for N requests or timeout, batch them (Triton)
  Continuous batching: Replace finished sequences immediately (vLLM, TRT-LLM)
  
Key Formulas:
  Arithmetic Intensity = batch_size × 2 (FLOPs per byte read)
  GPU Utilization ≈ Arithmetic_Intensity / (Peak_FLOPS / HBM_BW)
  Memory per request ≈ 2 × num_layers × hidden_dim × seq_len × bytes_per_param
```

---

*Next: [03 — vLLM, TensorRT-LLM, SGLang, llama.cpp Internals →](03_Inference_Engines_Deep_Dive.md)*
