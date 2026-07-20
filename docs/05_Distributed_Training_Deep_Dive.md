# Distributed Training: DDP, FSDP, ZeRO, DeepSpeed, Megatron-LM, Pipeline & Tensor Parallelism

> *Training an LLM requires more compute and memory than a single GPU, or even a single node, can provide. Distributed training is the art of breaking the model and the data apart, placing the pieces across thousands of GPUs, and keeping them synchronized over a high-speed network.*

---

## 1. What Problem Does This Solve?

### The Memory Wall

To train a 70B parameter model, you need memory for:
1. **Model Weights:** 70B × 2 bytes (FP16) = 140 GB
2. **Gradients:** 70B × 2 bytes (FP16) = 140 GB
3. **Optimizer States:** Adam requires a momentum and variance term for every parameter in FP32. 70B × 4 bytes × 2 = 560 GB
4. **Activations:** Intermediate outputs saved during the forward pass to compute gradients during the backward pass (varies by batch size, often 100+ GB)

**Total Memory Required to Train 70B:** ~1,000 GB (1 Terabyte)
**Max VRAM on a single H100:** 80 GB

It is physically impossible to train a 70B model on a single GPU. We must distribute the memory and compute.

### Distributed Training Paradigms

1. **Data Parallelism (DDP):** Model fits on 1 GPU. Copy the model to N GPUs. Give each GPU a different slice of data.
2. **Fully Sharded Data Parallelism (FSDP / ZeRO):** Model does NOT fit on 1 GPU. Shard the model, gradients, and optimizer states across N GPUs.
3. **Tensor Parallelism (TP):** Shard individual matrix operations across GPUs (requires extremely fast NVLink).
4. **Pipeline Parallelism (PP):** Put layers 1-10 on GPU 1, layers 11-20 on GPU 2, etc. (requires careful scheduling to avoid idle bubbles).

---

## 2. Internal Architecture

### PyTorch Distributed Data Parallel (DDP)

**How it works:**
1. Every GPU holds an identical copy of the entire model.
2. During the forward pass, each GPU processes a different micro-batch of data.
3. During the backward pass, each GPU computes its local gradients.
4. **The AllReduce Step:** Before updating the weights, all GPUs synchronize their gradients over the network using `Ring-AllReduce`. The average gradient is computed.
5. Every GPU applies the same average gradient, ensuring their model weights remain identical for the next step.

**Bottleneck:** Memory. It requires the entire model to fit on a single GPU.

### DeepSpeed ZeRO (Zero Redundancy Optimizer)

ZeRO attacks the memory wall by eliminating the redundancy in DDP. If you have 8 GPUs, why store 8 identical copies of the 560 GB Adam optimizer state?

```
┌─────────────────────────────────────────────────────────────┐
│                 ZeRO Memory Sharding                        │
│                                                             │
│  Baseline (DDP):                                            │
│  GPU 0: [Weights] [Gradients] [Optimizer]                   │
│  GPU 1: [Weights] [Gradients] [Optimizer]                   │
│                                                             │
│  ZeRO Stage 1 (Shard Optimizer State):                      │
│  GPU 0: [Weights] [Gradients] [Opt_Part0]                   │
│  GPU 1: [Weights] [Gradients] [Opt_Part1]                   │
│  (Saves 4x memory on 8 GPUs)                                │
│                                                             │
│  ZeRO Stage 2 (Shard Gradients):                            │
│  GPU 0: [Weights] [Grad_Part0] [Opt_Part0]                  │
│  GPU 1: [Weights] [Grad_Part1] [Opt_Part1]                  │
│  (Saves 8x memory on 8 GPUs)                                │
│                                                             │
│  ZeRO Stage 3 / FSDP (Shard Everything):                    │
│  GPU 0: [Weight_Part0] [Grad_Part0] [Opt_Part0]             │
│  GPU 1: [Weight_Part1] [Grad_Part1] [Opt_Part1]             │
│  (Saves N-x memory. Model size scales linearly with GPUs)   │
└─────────────────────────────────────────────────────────────┘
```

**How ZeRO-3 / FSDP computes the forward pass if it only has 1/8th of the weights:**
1. GPU 0 needs Weight_Part1. It requests it from GPU 1 via an `AllGather` operation over the network.
2. GPU 0 computes the layer.
3. GPU 0 immediately discards Weight_Part1 to free up memory.
4. It is a massive trade-off: **Save memory, but dramatically increase network traffic.**

### Megatron-LM (3D Parallelism)

When training massive models (e.g., GPT-3 175B), FSDP/ZeRO-3 generates too much network traffic for slow inter-node networks. You must combine three techniques (3D Parallelism):

1. **Tensor Parallelism (Intra-Node):** Split matrix multiplications across the 8 GPUs within a single physical node. Communication is over NVLink (900 GB/s), which is fast enough to hide the latency.
2. **Pipeline Parallelism (Inter-Node):** Split the model depth-wise across nodes. Node A does layers 1-10, Node B does 11-20. Communication is just passing activations over InfiniBand, which is lightweight.
3. **Data Parallelism (Global):** Copy this entire pipeline across hundreds of nodes to scale batch size.

---

## 3. Deep Internal Working

### Tensor Parallelism (Row & Column Sharding)

Imagine calculating `Y = X × W` where W is a massive weight matrix.

**Column Parallelism:**
Split W vertically.
GPU 0 computes `Y1 = X × W_left`
GPU 1 computes `Y2 = X × W_right`
To get the final Y, we concatenate `[Y1, Y2]`. (Requires an `AllGather`).

**Row Parallelism:**
Split W horizontally. Split X vertically.
GPU 0 computes `Y1 = X_left × W_top`
GPU 1 computes `Y2 = X_right × W_bottom`
To get the final Y, we add them: `Y = Y1 + Y2`. (Requires an `AllReduce`).

Because Tensor Parallelism requires an AllReduce/AllGather at **every single layer** of the transformer, it is incredibly sensitive to network latency. It should almost never cross physical node boundaries (e.g., never use TP=16 if you only have 8 GPUs per node).

### Pipeline Parallelism (The Bubble Problem)

```
Naive Pipeline:
GPU 1: [Layer 1]────>
GPU 2:              [Layer 2]────>
GPU 3:                           [Layer 3]────>
GPU 4:                                        [Layer 4]

Problem: While GPU 1 works, GPUs 2, 3, and 4 sit idle.
         This idle time is called the "Pipeline Bubble."

GPipe / 1F1B (One Forward, One Backward) Solution:
Break the micro-batch into even smaller chunks.
GPU 1: [F1][F2][F3][F4]
GPU 2:     [F1][F2][F3][F4]
GPU 3:         [F1][F2][F3][F4]
GPU 4:             [F1][F2][F3][F4][B1][B2][B3][B4]
GPU 3:                 [B1][B2][B3][B4]
GPU 2:                     [B1][B2][B3][B4]
GPU 1:                         [B1][B2][B3][B4]

By pipelining chunks, we shrink the bubble to ~10% of total execution time.
```

---

## 4. Production Architecture

### Enterprise Training Cluster

```
┌────────────────────────────────────────────────────────────────────┐
│                    Large Scale Training Architecture               │
│                                                                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐       │
│  │ Node 0 (8xH100) │ │ Node 1 (8xH100) │ │ Node N (8xH100) │       │
│  │ TP Rank: 0-7    │ │ TP Rank: 0-7    │ │ TP Rank: 0-7    │       │
│  │ PP Rank: 0      │ │ PP Rank: 1      │ │ PP Rank: M      │       │
│  │ DP Rank: 0      │ │ DP Rank: 0      │ │ DP Rank: Z      │       │
│  └──────┬──────────┘ └──────┬──────────┘ └──────┬──────────┘       │
│         │                   │                   │                  │
│  ┌──────┴───────────────────┴───────────────────┴────────────┐     │
│  │          InfiniBand NDR Fabric (Non-blocking)             │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Job Orchestrator: Kubernetes (Volcano) or Slurm                   │
│  Storage: Weka / Lustre / GPFS                                     │
│  Checkpoints: S3 via high-speed object connector                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. Production Incident Scenarios

### Incident 1: "NCCL Timeout / GPU Hang"
**Symptoms:** Training runs for 2 hours, then hangs indefinitely. No errors. `nvidia-smi` shows 100% Volatile GPU Util, but `dcgmi` shows 0% SM Util.
**Root Cause:** A node experienced a slight PCIe bus error or thermal throttle, causing it to run 500ms slower than the others. The other 255 GPUs reached the `AllReduce` synchronization barrier and waited. Because the slow GPU never arrived (or dropped a network packet), the NCCL operation timed out.
**Fix:** Set `NCCL_ASYNC_ERROR_HANDLING=1` to crash the job instead of hanging. Use `NCCL_DEBUG=INFO` to identify which rank caused the hang. Restart from the last checkpoint.

### Incident 2: "Straggler Node Halves Training Speed"
**Symptoms:** Global steps/sec drops by 50%.
**Root Cause:** In synchronous distributed training, the entire cluster runs exactly as fast as the **slowest GPU**. One GPU in a 1,024 GPU cluster had a degraded cooling fan, throttling its clock speed by 50%. It dragged the entire cluster down.
**Fix:** Implement a "straggler detection" script that profiles step times per rank. Evict the bad node and resume training.

### Incident 3: "OOM During Checkpoint Save"
**Symptoms:** Training completes the epoch, attempts to save the checkpoint, and OOMs.
**Root Cause:** In FSDP, the weights are sharded across all GPUs. To save a standard PyTorch checkpoint, Rank 0 must issue an `AllGather` to pull all shards into its own RAM before writing to disk. This caused an OOM on Rank 0.
**Fix:** Save distributed checkpoints. Have each GPU write its own shard to disk independently, and stitch them together offline if needed.

---

## 6. Performance Optimization

### The Batch Size vs Communication Trade-off
- **Compute bound:** The GPUs are spending their time doing matrix math.
- **Communication bound:** The GPUs are spending their time waiting for NCCL `AllReduce` over the network.

To fix communication bottlenecks, increase the **Global Batch Size** and implement **Gradient Accumulation**.
```python
# Instead of AllReduce every step:
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()  # Accumulate local gradients
    
    # Only communicate over network every 4 steps
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 7. Kubernetes Perspective (PyTorchJob)

Running a distributed job on Kubernetes requires the **Kubeflow Training Operator**. It manages the headless services and environment variables (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`) required by PyTorch DDP.

```yaml
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "llama-finetune"
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training-image
            command: ["torchrun", "--nnodes=4", "--nproc_per_node=8", "train.py"]
            resources:
              limits:
                nvidia.com/gpu: 8
    Worker:
      replicas: 3  # Total 4 nodes (1 master + 3 workers)
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training-image
            resources:
              limits:
                nvidia.com/gpu: 8
```

---

## Summary

```
Algorithm Selection Cheat Sheet:
  Fits on 1 GPU?             → Use DDP
  Fits on 1 Node (8 GPUs)?   → Use FSDP (ZeRO-3) or Tensor Parallelism
  Cluster of Nodes?          → 
    Small/Medium Model       → FSDP across nodes
    Massive Model (>70B)     → 3D Parallelism (Megatron-LM: TP within node, PP across nodes, DP across clusters)

Key Network Dependencies:
  DDP:    Bandwidth hungry (syncs massive gradients)
  FSDP:   Bandwidth VERY hungry (syncs weights forward, gradients backward)
  TP:     Latency critical (syncs every layer, requires NVLink)
  PP:     Latency tolerant (only passes activations between stages)
```

---
*Next: [06 — Model Serving (Triton, vLLM, KServe) →](06_Model_Serving_Deep_Dive.md)*
