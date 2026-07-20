# Multi-GPU & Multi-Node Inference

> *While distributed training is well-understood, distributed inference is the new frontier. When serving a 400B parameter model, a single node's memory is exhausted before you even load the KV cache. You must split the model across GPUs and Nodes, introducing latency that must be aggressively hidden.*

---

## 1. What Problem Does This Solve?

### The VRAM Constraint

Consider LLaMA-3.1 405B (FP16):
- **Model weights:** 405B × 2 bytes = **810 GB**
- **Max VRAM on an 8x H100 Node:** 8 × 80 GB = **640 GB**

*You literally cannot load this model on a single $300,000 server.*

Even with INT8 quantization (405 GB), loading it on one node leaves only 235 GB for the KV cache. For a production endpoint with thousands of users and 128k context windows, you need terabytes of KV cache memory.

### Alternatives

1. **Wait for larger GPUs:** H200 has 141GB. B200 has 192GB. But model sizes are growing faster than hardware memory capacity.
2. **Aggressive Quantization (INT4):** Shrinks 405B to ~210GB. Fits on one node easily, but sacrifices reasoning quality.
3. **CPU Offloading:** Store weights in System RAM (2TB+) and stream to GPU over PCIe. *Result: Unacceptably slow (1 token every 3 seconds) due to PCIe Gen5 bandwidth limits (128 GB/s).*

**The Preferred Solution:** Shard the model across multiple GPUs (Tensor Parallelism) and, if necessary, multiple physical nodes (Pipeline Parallelism).

---

## 2. Internal Architecture

### Tensor Parallelism (TP) for Inference

Tensor Parallelism splits individual transformer layers (Linear projections and Attention heads) across GPUs.

```
┌────────────────────────────────────────────────────────────────┐
│                   Tensor Parallel Inference (TP=4)             │
│                                                                │
│  Input Token                                                   │
│       │                                                        │
│       ├──────────────┬──────────────┬──────────────┐           │
│       ▼              ▼              ▼              ▼           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │  GPU 0  │    │  GPU 1  │    │  GPU 2  │    │  GPU 3  │      │
│  │         │    │         │    │         │    │         │      │
│  │ W_QKV_0 │    │ W_QKV_1 │    │ W_QKV_2 │    │ W_QKV_3 │      │
│  │ Heads1-8│    │Heads9-16│    │Heads17-24    │Heads25-32      │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘      │
│       │              │              │              │           │
│       └──────────────┴──────┬───────┴──────────────┘           │
│                             ▼                                  │
│                   All-Reduce (via NVLink)                      │
│                (Synchronize partial results)                   │
│                             │                                  │
│                             ▼                                  │
│                       Next Layer                               │
└────────────────────────────────────────────────────────────────┘
```

**Why TP is fast:** Every GPU computes its fraction of the math simultaneously. The All-Reduce over NVLink (900 GB/s) takes microseconds. The total time to process a layer actually *decreases* as you add GPUs.

### Pipeline Parallelism (PP) for Inference

When you cross a node boundary, NVLink is gone. You are on InfiniBand (or RoCE). An All-Reduce at every layer over IB is too slow for inference latency requirements.

Instead, we use Pipeline Parallelism: Node A computes layers 1-40, Node B computes layers 41-80.

```
┌────────────────────────────────────────────────────────┐
│              Pipeline Parallel Inference (PP=2)        │
│                                                        │
│  ┌─────────────────┐           ┌─────────────────┐     │
│  │     Node A      │           │     Node B      │     │
│  │  (Layers 1-40)  │           │  (Layers 41-80) │     │
│  │                 │           │                 │     │
│  │ Req 1 (Token 1) │── IB ───> │ Req 1 (Token 1) │     │
│  │ Req 2 (Token 1) │── IB ───> │ Req 2 (Token 1) │     │
│  │                 │           │                 │     │
│  └─────────────────┘           └─────────────────┘     │
└────────────────────────────────────────────────────────┘
```

**The Catch:** Node B is completely idle while Node A is processing. To fix this, you must run *continuous batching* and pass requests in a stream so both nodes are constantly working.

---

## 3. Deep Internal Working

### Memory Layout in vLLM (TP=4)

When you launch `vLLM` with `--tensor-parallel-size 4`, it spawns 4 worker processes via Ray or Multiprocessing.

1. **Weight Sharding:** The master process reads the safetensors. It slices `q_proj` (e.g., 8192 x 8192) into four 8192 x 2048 chunks and sends one chunk to each GPU.
2. **KV Cache Sharding:** The KV cache is also sharded. GPU 0 only stores the K and V vectors for attention heads 1-8. GPU 1 stores heads 9-16.
   *This is magical: As you increase TP, you don't just get more compute, you get linearly more KV cache capacity per request.*
3. **Execution:** The master API server receives a request. It broadcasts the token IDs to all 4 GPUs. They execute their kernels, synchronize via PyTorch Distributed (NCCL), and the master returns the result.

### NCCL Custom All-Reduce Kernels

Standard NCCL All-Reduce is designed for huge training tensors (100MB+). Inference tensors are tiny (e.g., batch_size 16 × hidden_dim 8192 = a few megabytes).

To minimize latency, inference engines use **Custom All-Reduce Kernels** that bypass the standard NCCL API overhead and write directly to peer GPU memory using NVLink hardware pointers. This reduces synchronization latency from ~15μs to ~2μs per layer.

---

## 4. Production Architecture

### Global Routing + Multi-Node vLLM (Llama-3 405B)

```
┌───────────────────────────────────────────────────────────────────────┐
│                    Llama-3 405B Production Serving                    │
│                                                                       │
│  ┌───────────────────────┐                                            │
│  │  Global Load Balancer │ (Nginx / Envoy)                            │
│  └───────────┬───────────┘                                            │
│              │                                                        │
│  ┌───────────▼───────────┐    ┌───────────▼───────────┐               │
│  │   Inference Replica 1 │    │   Inference Replica 2 │               │
│  │   (Capacity: 50 req/s)│    │   (Capacity: 50 req/s)│               │
│  │                       │    │                       │               │
│  │ ┌────────┐ ┌────────┐ │    │ ┌────────┐ ┌────────┐ │               │
│  │ │ Node A │ │ Node B │ │    │ │ Node C │ │ Node D │ │               │
│  │ │ L 1-63 │ │ L 64-126││    │ │ L 1-63 │ │ L 64-126││               │
│  │ │ (TP=8) │ │ (TP=8) │ │    │ │ (TP=8) │ │ (TP=8) │ │               │
│  │ └────┬───┘ └────┬───┘ │    │ └────┬───┘ └────┬───┘ │               │
│  │      └─── IB ───┘     │    │      └─── IB ───┘     │               │
│  │      (PP=2)           │    │      (PP=2)           │               │
│  └───────────────────────┘    └───────────────────────┘               │
└───────────────────────────────────────────────────────────────────────┘
```

**Deployment Spec:**
- `tensor-parallel-size = 8` (Max out the NVLink domain on the node)
- `pipeline-parallel-size = 2` (Split model across 2 nodes)
- Total GPUs per replica = 16

---

## 5. Production Incident Scenarios

### Incident 1: "Throughput Halves on Multi-Node"
**Symptoms:** 70B model on 1 Node (TP=8) gets 2,000 tok/s. 70B model on 2 Nodes (PP=2, TP=4 per node) gets 900 tok/s.
**Root Cause:** The pipeline bubble. With PP=2, a request must traverse Node A, go over the network, traverse Node B, and wait for the final token to be projected back to Node A. Unless the batch size is massive, the network latency of IB kills the throughput.
**Fix:** Avoid Pipeline Parallelism for inference unless absolutely necessary for memory constraints. Always max out Tensor Parallelism within a single node first.

### Incident 2: "NCCL Timeout during Decode"
**Symptoms:** Request generates 50 tokens normally, then hangs and times out.
**Root Cause:** A cosmic ray flipped a bit in the ECC memory of GPU 3, causing a minor hardware reset. GPU 3 paused for 100ms to recover. GPUs 0, 1, and 2 hit the All-Reduce barrier and waited. Because inference engines disable NCCL async error handling to maximize speed, the entire process hung permanently.
**Fix:** Configure liveness probes in K8s to query the `/health` endpoint of vLLM. If vLLM detects a hung worker, it will fail the probe, and K8s will restart the pod.

### Incident 3: "KV Cache OOM on TP=2 but not TP=4"
**Symptoms:** Model runs fine on 4 GPUs. When trying to save money by running on 2 GPUs (with quantization to fit the weights), it OOMs immediately.
**Root Cause:** Tensor Parallelism divides BOTH the model weights AND the KV Cache across GPUs. By dropping from TP=4 to TP=2, each GPU now has to store twice as much KV Cache state. The VRAM is exhausted.
**Fix:** Reduce `max_num_seqs` or lower `gpu_memory_utilization` to account for the increased KV cache burden per GPU.

---

## 6. Performance Optimization

### The TP vs PP Decision Matrix

1. **Rule 1:** NEVER cross a node boundary with Tensor Parallelism (TP). The lack of NVLink will destroy performance.
2. **Rule 2:** Maximize TP within a node (up to 8 on HGX systems). Higher TP = lower latency per token (Time Per Output Token - TPOT).
3. **Rule 3:** Only use Pipeline Parallelism (PP) when the model (Weights + KV Cache) exceeds the total VRAM of a single node.
4. **Rule 4:** If the model fits on 4 GPUs, run **two independent replicas** of TP=4 on your 8-GPU node, rather than one replica of TP=8. (Two independent replicas yield higher overall cluster throughput, while TP=8 yields lower latency for a single user).

---

## 7. Kubernetes Perspective

### Scheduling Multi-Node Inference Pods (Ray / KubeRay)

You cannot schedule a multi-node inference job with a standard Deployment. You must use a framework that understands worker coordination, like Ray.

```yaml
# KubeRay RayService Definition for multi-node vLLM
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: vllm-llama405b
spec:
  serveConfigV2: |
    applications:
      - name: llm
        import_path: vllm.engine.ray_utils:RayServeWrapper
        route_prefix: /
  rayClusterConfig:
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            resources: { requests: { cpu: "8" } }
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 2   # Two physical nodes
      template:
        spec:
          containers:
          - name: ray-worker
            resources:
              limits:
                nvidia.com/gpu: 8  # 8 GPUs per node (Total 16 GPUs)
```
*vLLM will automatically detect the Ray cluster and distribute the PP and TP ranks across the workers.*

---

## Summary

```
Distributed Inference Cheat Sheet:

| Parallelism | Domain | Network | Impact on Latency | Impact on Throughput |
|---|---|---|---|---|
| **Tensor (TP)** | Intra-Node | NVLink | Decreases Latency | Neutral |
| **Pipeline (PP)**| Inter-Node | IB/RoCE | Increases Latency | Neutral (enables larger models) |
| **Data (DP)** | Global | HTTP | Neutral | Increases Throughput |

Memory Math:
Total VRAM required = (Weights / Quantization_factor) + (KV_Cache_per_req × Max_Reqs)
If Total VRAM > Node VRAM → You MUST use Pipeline Parallelism.
```

---
*Next: [10 — LoRA, QLoRA, PEFT, and Distributed Fine-Tuning →](10_LoRA_FineTuning_Pipelines_Deep_Dive.md)*
