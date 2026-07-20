# Model Serving: NVIDIA Triton, vLLM, KServe, Ray Serve, SGLang, TorchServe

> *Training an AI model is a data science problem. Serving an AI model in production is a distributed systems engineering problem. The choice of serving framework dictates your SLA, your infrastructure cost, and your operational overhead.*

---

## 1. What Problem Does This Solve?

### The Serving Challenge

You have a trained model weight file (`.pt`, `.safetensors`, `.onnx`). You need to serve it to 10,000 concurrent users with < 100ms latency.

Naive approach (Flask + PyTorch):
```python
@app.route("/predict")
def predict():
    # 1. User waits in HTTP queue
    tensor = preprocess(request.json)
    # 2. GPU processes ONE request at a time
    result = model(tensor)
    return postprocess(result)
```

**Why this fails in production:**
1. **No Hardware Utilization:** A GPU processing batch size 1 is 99% idle.
2. **GIL Blocking:** Python's Global Interpreter Lock prevents true multi-threading.
3. **No GPU Sharing:** You can't safely load an LLM and an embedding model on the same GPU via two Flask apps (CUDA OOM crashes).
4. **No Autoscaling:** The app doesn't know how to scale based on GPU utilization.

**Model Serving Frameworks solve this by providing:**
- C++ execution engines (bypassing the Python GIL)
- Dynamic batching (grouping requests over a time window)
- Multi-model serving (bin-packing models onto one GPU safely)
- Metrics (Prometheus integration)
- Standardized APIs (gRPC, HTTP/REST)

---

## 2. Framework Comparison & Internal Architecture

### NVIDIA Triton Inference Server (The Enterprise Standard)

Triton is a C++ server designed to maximize hardware utilization across any framework (TensorFlow, PyTorch, ONNX, TensorRT).

**Architecture:**
```
┌──────────────────────────────────────────────────────────────────┐
│                   NVIDIA Triton Inference Server                 │
│                                                                  │
│  ┌─────────────────┐ ┌───────────────┐ ┌──────────────────────┐  │
│  │ HTTP/REST (8000)│ │ gRPC (8001)   │ │ C API (in-process)   │  │
│  └───────┬─────────┘ └───────┬───────┘ └──────────┬───────────┘  │
│          └───────────────────┼────────────────────┘              │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   Triton Core Scheduler                    │  │
│  │  • Dynamic Batcher                                         │  │
│  │  • Sequence Batcher (for stateful models)                  │  │
│  │  • Ensemble Scheduler (pipelines)                          │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                       Backend API                          │  │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │  │
│  │ │ TensorRT│ │ ONNX RT │ │ PyTorch │ │ Python  │ │ vLLM   │ │  │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                       Hardware                             │  │
│  │  [ GPU 0 (A100) ]      [ GPU 1 (A100) ]       [ CPU ]      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Key Feature - Dynamic Batching:**
Triton intercepts incoming requests, holds them in a queue for a configurable window (e.g., 10ms), concatenates the tensors along the batch dimension, sends them to the GPU, and then scatters the results back to the individual HTTP clients.

### vLLM & SGLang (The LLM Specialists)

*(See Chapter 3 for deep architectural dive into vLLM/SGLang)*

While Triton is general-purpose, vLLM is purpose-built for autoregressive text generation. Triton has integrated vLLM as a backend, allowing you to use Triton's gRPC endpoints and metrics with vLLM's PagedAttention engine.

### KServe (The Kubernetes Native Orchestrator)

KServe is not an inference engine (like vLLM) — it is a **Kubernetes Custom Resource Definition (CRD)** that orchestrates inference engines.

It wraps your inference engine (Triton, vLLM) in a Knative serverless container, providing:
- Scale-to-zero
- Traffic splitting (Canary deployments, A/B testing)
- Request routing

**Architecture:**
```yaml
# KServe InferenceService definition
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "llama-3-model"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    model:
      modelFormat:
        name: vLLM
      storageUri: "s3://my-bucket/models/llama-3"
      resources:
        limits:
          nvidia.com/gpu: 1
```

### Ray Serve (The Distributed Python Orchestrator)

Ray Serve is built on top of Ray (a distributed computing framework). It excels at **model composition** (Ensembles / Pipelines).

If your pipeline is: `Image Preprocessing (CPU) -> ViT Model (GPU) -> Text Postprocessing (CPU)`, Ray Serve allows you to deploy these components to different nodes in a cluster and handles the RPC routing between them.

---

## 3. Deep Internal Working: Memory Management & Zero-Copy

### The IPC Problem (Inter-Process Communication)

If you use Python (FastAPI) to receive an image, convert it to a NumPy array, and send it to a backend GPU server, that data is copied multiple times:
1. Network Buffer → FastAPI memory
2. FastAPI memory → NumPy array (CPU RAM)
3. NumPy array → gRPC Protobuf serialization (CPU RAM)
4. gRPC Protobuf → Server memory (CPU RAM)
5. Server memory → GPU VRAM (via PCIe DMA)

This overhead destroys performance for high-throughput systems.

### Triton's Zero-Copy Shared Memory

Triton supports **CUDA Shared Memory**. A client process running on the same node can allocate a block of CUDA memory, write the input tensor directly to it, and simply send a pointer (memory handle) to Triton via gRPC.

```python
# Triton CUDA Shared Memory (Zero-Copy)
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm

# 1. Allocate VRAM directly
shm_handle = shm.create_shared_memory_region("my_input_shm", byte_size, gpu_id=0)

# 2. Write data directly to VRAM
shm.set_shared_memory_region(shm_handle, [input_tensor])

# 3. Tell Triton to run inference using the pointer (NO DATA TRANSFER)
client.infer(model_name="resnet", inputs=[grpcclient.InferInput(..., "my_input_shm")])
```

---

## 4. Production Architecture

### Enterprise Gateway + LLM Serving (The Multi-Model Pattern)

```
┌─────────────────────────────────────────────────────────────┐
│                 Enterprise AI Platform                      │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ AI Gateway      │ (Auth, Rate Limiting, Audit Logging)   │
│  │ (Kong/Envoy)    │                                        │
│  └───────┬─────────┘                                        │
│          │                                                  │
│  ┌───────▼─────────┐                                        │
│  │ Semantic Router │ (Routes based on task difficulty)      │
│  └───────┬─────────┘                                        │
│          │                                                  │
│  ┌───────┴───────────────────────────────┐                  │
│  │                                       │                  │
│  ▼                                       ▼                  │
│ ┌───────────────────┐               ┌───────────────────┐   │
│ │ KServe Predictor  │               │ KServe Predictor  │   │
│ │ (Triton Backend)  │               │ (vLLM Backend)    │   │
│ │ Embeddings        │               │ Llama-3-70B       │   │
│ │ 2x A10G GPUs      │               │ 8x H100 GPUs      │   │
│ └───────────────────┘               └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Production Incident Scenarios

### Incident 1: "Triton Python Backend Hangs"
**Symptoms:** Requests to a custom Python model in Triton timeout. Throughput is stuck at 1 req/sec.
**Root Cause:** The Python backend executes in a separate process. The developer forgot to set `KIND_GPU` and the model ran on CPU, or the developer used a blocking HTTP call inside the `execute()` function, blocking the entire Triton instance.
**Fix:** Move blocking I/O out of the Triton backend (put it in the API gateway). Ensure PyTorch models use the TensorRT or LibTorch (C++) backends, not the Python backend.

### Incident 2: "Scale-to-Zero Causes 45 Second Cold Starts"
**Symptoms:** Users complain that the first request of the day times out.
**Root Cause:** KServe was configured with `minReplicas: 0`. When a request arrives, KNative spins up a pod. The pod must download 140GB of weights from S3 to the node, then load them from disk into GPU VRAM. This takes 45-60 seconds.
**Fix:** Set `minReplicas: 1` for large LLMs. Alternatively, use **Storage Tiering/Host Caching** (e.g., dataset PVCs) so the weights are already on the NVMe drive of the node, bypassing the network download.

### Incident 3: "CUDA OOM on Triton Multi-Model"
**Symptoms:** Triton crashes with CUDA OOM when loading the 3rd model.
**Root Cause:** Triton loads models into VRAM concurrently. Model A takes 10GB, Model B takes 10GB. The GPU has 24GB. When Model C (10GB) tries to load, the GPU OOMs.
**Fix:** Set Triton memory limits, or use Kubernetes GPU requests to bin-pack correctly. Alternatively, compile the models to TensorRT, which allows you to specify exact workspace memory limits at build time.

---

## 6. Performance Optimization

### Triton Dynamic Batching Tuning

```pbtxt
# config.pbtxt (Triton configuration file)
name: "resnet50"
platform: "tensorrt_plan"
max_batch_size: 128

dynamic_batching {
  # Wait up to 50ms for more requests to form a batch
  max_queue_delay_microseconds: 50000
  
  # Triton will try to form batches of exactly these sizes
  # (Useful for TensorRT engines optimized for specific shapes)
  preferred_batch_size: [ 16, 32, 64, 128 ]
}
```

### TensorRT Compilation (The Ultimate Optimization)

Never serve raw `.pt` or `.onnx` files in high-throughput production. Always compile to TensorRT (`.engine`).
TensorRT performs:
- Precision calibration (INT8)
- Layer fusion (fusing activation functions into convolutions)
- Kernel auto-tuning (running thousands of kernels on the *actual target GPU* during the build phase to find the fastest one)

---

## Summary

```
Framework Selection Guide:

| Framework | Best For | Core Strength | Weakness |
|---|---|---|---|
| **vLLM** | Text Generation (LLMs) | PagedAttention, Speed | Only for generative text |
| **Triton** | CV, Embeddings, Ensembles | Dynamic Batching, C++ | Steep learning curve |
| **KServe** | K8s Orchestration | Scale-to-zero, Canary | Complex Istio/Knative stack |
| **Ray Serve**| Model Pipelines | Python-native RPC | Heavy infrastructure |
| **SGLang** | Structured Gen (JSON) | RadixAttention | Newer, less enterprise support|
```

---
*Next: [07 — Kubernetes GPU Orchestration (MIG, Device Plugin) →](07_GPU_Kubernetes_Orchestration.md)*
