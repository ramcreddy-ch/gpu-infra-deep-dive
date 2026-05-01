# 07. LLMOps: Serving Large Language Models at Scale

LLM serving is not just "model serving but bigger." It introduces an entirely new set of challenges around memory management, token streaming, quantization trade-offs, and prompt engineering at the infrastructure level. This chapter covers everything I've learned running LLM inference endpoints that serve thousands of concurrent users.

**Author:** [Ramchandra Chintala](https://github.com/ramcreddy-ch)

---

## Why LLM Serving Is Different

Traditional ML serving (like a fraud detection classifier) takes a fixed-size input and returns a fixed-size output in one shot. An LLM, on the other hand:

1. **Generates tokens one at a time** (autoregressive), so latency compounds with output length.
2. **Maintains state between tokens** via the KV Cache, which grows with context length and eats vRAM.
3. **Has wildly variable request sizes.** One user sends "Hi", the next sends a 50-page document. Your serving infrastructure has to handle both without crashing.

### The Math That Matters

Before deploying any LLM, I do this back-of-the-envelope calculation:

```
Model Memory (FP16) = Parameters × 2 bytes
                    = 7B × 2 = 14 GB

KV Cache per request = 2 × num_layers × hidden_size × context_length × 2 bytes
                     = 2 × 32 × 4096 × 4096 × 2 = ~2 GB per concurrent user

Total for 50 concurrent users = 14 GB + (50 × 2 GB) = 114 GB
```

That's more than a single A100 (80 GB). This is why you see LLM endpoints using tensor parallelism across multiple GPUs even for a "small" 7B model when serving real traffic.

---

## The LLM Serving Stack

### vLLM: My Go-To for Production

After trying TGI, Triton, and vanilla PyTorch, I settled on vLLM for most production workloads. Here's why:
- **PagedAttention** eliminates KV Cache memory waste (covered in Chapter 05)
- **Continuous batching** dynamically groups requests instead of waiting for a full batch
- **Tensor parallelism** splits the model across GPUs with minimal code changes

```yaml
# vLLM deployment on K8s
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-7b
  namespace: llm-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-llama-7b
  template:
    metadata:
      labels:
        app: vllm-llama-7b
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model=meta-llama/Llama-2-7b-chat-hf"
            - "--tensor-parallel-size=1"
            - "--max-model-len=4096"
            - "--gpu-memory-utilization=0.90"
            - "--max-num-batched-tokens=8192"
            - "--enable-prefix-caching"
            - "--quantization=awq"
            - "--port=8000"
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "64Gi"
              cpu: "16"
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
            - name: model-cache
              mountPath: /root/.cache/huggingface
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 120
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 10
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "16Gi"
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      nodeSelector:
        workload-type: gpu-serving
```

**Flags I always tune:**
- `--gpu-memory-utilization=0.90` — Reserve 10% vRAM headroom for spikes. Setting this to 0.95+ causes sporadic OOMs under load.
- `--enable-prefix-caching` — If users send similar system prompts, vLLM caches the KV computations. Saves ~30% compute for chatbot workloads.
- `--quantization=awq` — AWQ quantization drops a 7B model from 14GB to ~4GB with minimal quality loss. Game changer for fitting on smaller GPUs.

### NVIDIA Triton: When You Need Multi-Framework Support

If you're serving a mix of PyTorch, TensorFlow, ONNX, and TensorRT models from the same endpoint, Triton is the right choice. I use it when the team has legacy TF models alongside newer PyTorch ones.

```yaml
# Triton model repository structure
model-repository/
├── fraud_detector/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
├── llama_7b/
│   ├── 1/
│   │   └── model.plan  # TensorRT compiled
│   └── config.pbtxt
└── embedding_model/
    ├── 1/
    │   └── model.pt
    └── config.pbtxt
```

---

## Quantization: The Art of Making Models Smaller

In production, I almost never serve models in full FP16. The cost/performance trade-off of quantization is too good to ignore.

### Quantization Comparison (LLaMA 7B)

| Method | Size | vRAM | Tokens/sec (A100) | Quality Loss |
|--------|------|------|--------------------|-------------|
| FP16 (baseline) | 14 GB | 16 GB | 45 t/s | None |
| GPTQ (4-bit) | 4 GB | 6 GB | 120 t/s | ~1-2% on benchmarks |
| AWQ (4-bit) | 3.9 GB | 5.5 GB | 130 t/s | <1% on benchmarks |
| GGUF (4-bit, CPU) | 4 GB | 0 GB (CPU) | 15 t/s | ~2-3% |
| FP8 (H100 only) | 7 GB | 9 GB | 90 t/s | <0.5% |

**My rule of thumb:**
- **Training:** Always FP16 or BF16. Never quantize during training.
- **Inference (latency-sensitive):** AWQ 4-bit on GPU. Best speed-to-quality ratio I've seen.
- **Inference (cost-sensitive):** GPTQ 4-bit. Slightly worse throughput but more model support.
- **Edge/CPU inference:** GGUF with llama.cpp. No GPU needed at all.

---

## Prompt Routing and Gateway Architecture

When you're serving multiple LLMs (small for simple queries, large for complex reasoning), you need a smart gateway:

```
User Request
     │
     ▼
┌──────────────┐
│  LLM Gateway │ ── Classifies prompt complexity
│  (FastAPI)   │ ── Checks rate limits
│              │ ── Adds system prompts
└──────┬───────┘
       │
       ├── Simple query ──► Llama 7B (fast, cheap)
       ├── Complex query ──► Llama 70B (slow, expensive)
       └── Code query ──► CodeLlama 34B (specialized)
```

I built a lightweight gateway that routes based on estimated token count and prompt category. This alone cut our GPU costs by 40% because 70% of user queries are simple enough for the smallest model.

---

## Real-Time LLMOps Issues I've Dealt With

### 1. The "Infinite Generation" Problem
A user sends a prompt that causes the model to loop endlessly, generating thousands of tokens and blocking the GPU for minutes.
**Fix:** Always set `--max-tokens` at the server level (not just client-side). vLLM's `--max-model-len` caps total context, but you also need per-request limits in your API gateway.

### 2. KV Cache Exhaustion Under Load
During a traffic spike, the KV cache fills up and new requests start getting rejected with 503 errors.
**Fix:** Configure vLLM's `--max-num-seqs` to limit concurrent sequences. Better to queue requests gracefully than to OOM the entire server. I also set up KEDA autoscaling based on the queue depth metric.

### 3. Model Loading Takes 5+ Minutes
When a new pod scales up, it downloads and loads a 14GB model from S3. During this time, the pod is "Running" but not ready, and health checks fail.
**Fix:** Use a PersistentVolumeClaim (PVC) to cache model weights across pod restarts. The `model-cache-pvc` in my deployment above ensures the model is only downloaded once per node.

### 4. Token Streaming Disconnects
Users expect streaming responses (like ChatGPT). But Kubernetes Ingress controllers and load balancers often have short timeout defaults that kill long-running SSE connections.
**Fix:** Configure your Ingress annotations:
```yaml
nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
nginx.ingress.kubernetes.io/proxy-buffering: "off"
```

### 5. Multi-GPU Tensor Parallelism Failures
When splitting a 70B model across 8 GPUs with `--tensor-parallel-size=8`, NCCL sometimes hangs during the initial all-gather.
**Fix:** Ensure all 8 GPUs are on the same physical node (use `nodeSelector`). Cross-node tensor parallelism over the network is possible but significantly slower. Also set `NCCL_P2P_DISABLE=0` to enable direct GPU-to-GPU NVLink communication.

---

*Next: [08. Day-to-Day GPU Operations & War Stories](./08_Day_to_Day_GPU_Operations.md)*
