# LoRA, QLoRA, PEFT & Distributed Fine-Tuning Pipelines

> *Full-parameter fine-tuning of a 70B model requires a multi-million dollar cluster and terabytes of VRAM. Parameter-Efficient Fine-Tuning (PEFT) allows you to customize massive models on a single GPU node. This is how enterprise AI adapts base models to domain-specific tasks without bankrupting the IT department.*

---

## 1. What Problem Does This Solve?

### The Fine-Tuning Memory Crisis

If you want to fine-tune LLaMA-3 70B using standard Full-Parameter tuning, you must load:
1. **Model Weights (FP32/BF16):** 140 GB
2. **Gradients (FP32/BF16):** 140 GB
3. **Adam Optimizer States:** 280 GB (Momentum) + 280 GB (Variance) = 560 GB
4. **Activations:** ~100 GB

**Total VRAM needed:** ~940 GB.
This requires at least 12-16 A100 (80GB) GPUs, plus a high-speed InfiniBand network to synchronize gradients (ZeRO-3 / FSDP).

### The PEFT Solution

What if, instead of updating all 70 billion parameters, we freeze the original model and only train a tiny "adapter" module containing a few million parameters?

1. **LoRA (Low-Rank Adaptation):** Injects trainable rank-decomposition matrices into transformer layers.
2. **QLoRA:** Quantizes the frozen base model to 4-bit (NF4) to save massive VRAM, while training a 16-bit LoRA adapter on top.

**Result:** You can fine-tune a 70B model on **two A100 GPUs (or one H100)**. A 7B model can be fine-tuned on a **single consumer RTX 4090 (24GB)**.

---

## 2. Internal Architecture

### LoRA Architecture (The Math)

Imagine a pre-trained weight matrix $W$ in the attention layer (e.g., the Query projection matrix) with dimensions $d \times d$ (e.g., 4096 $\times$ 4096).
During full fine-tuning, we update $W$ by adding a gradient matrix $\Delta W$ of the same size (16 million parameters).

**LoRA hypothesis:** The actual "learning" happens in a lower dimensional space (low intrinsic rank).
Instead of training $\Delta W$, we decompose it into two tiny matrices, $A$ and $B$, with a small rank $r$ (e.g., $r=8$).

```
Original Forward Pass:
  h = Wx

LoRA Forward Pass:
  h = Wx + (BA)x

Where:
  W is frozen (4096 × 4096) = 16,777,216 parameters
  A is trainable (r × 4096) = 8 × 4096 = 32,768 parameters
  B is trainable (4096 × r) = 4096 × 8 = 32,768 parameters
  
Total Trainable Parameters: 65,536 (99.6% reduction!)
```

### QLoRA Architecture (The Hardware Perspective)

QLoRA (Quantized LoRA) takes this further by compressing the frozen base model $W$ into 4-bit NormalFloat (NF4).

```
┌─────────────────────────────────────────────────────────────┐
│                       QLoRA Execution                       │
│                                                             │
│                 Input Token (x) (FP16)                      │
│                           │                                 │
│          ┌────────────────┴────────────────┐                │
│          ▼                                 ▼                │
│  ┌────────────────┐                ┌────────────────┐       │
│  │ Base Model (W) │                │ LoRA Adapter   │       │
│  │ (Frozen 4-bit  │                │ (Trainable)    │       │
│  │  NF4 memory)   │                │                │       │
│  │                │                │ ┌────────────┐ │       │
│  │ *Dequantize to │                │ │ Matrix A   │ │       │
│  │  FP16 on fly   │                │ │ (FP16/BF16)│ │       │
│  │ *MatMul (FP16) │                │ └──────┬─────┘ │       │
│  └───────┬────────┘                │        ▼       │       │
│          │                         │ ┌────────────┐ │       │
│          │                         │ │ Matrix B   │ │       │
│          │                         │ │ (FP16/BF16)│ │       │
│          │                         │ └──────┬─────┘ │       │
│          │                         └────────│───────┘       │
│          └────────────────┬─────────────────┘               │
│                           ▼                                 │
│                   Addition (FP16)                           │
│                           │                                 │
│                    Output (FP16)                            │
└─────────────────────────────────────────────────────────────┘
```
*Note: The frozen weights are stored in 4-bit but computed in 16-bit. The adapter is stored and computed in 16-bit. Paged Optimizers are used to offload optimizer states to CPU RAM if GPU VRAM spikes.*

---

## 3. Deep Internal Working

### Paged Optimizers

During QLoRA, if the context length is large, the activations can still cause an OOM. QLoRA introduces **Paged Optimizers** which leverage NVIDIA Unified Memory. If the GPU runs out of VRAM, the optimizer states (Adam momentum) are automatically evicted to host CPU RAM over PCIe, and paged back in when needed.

### Merging LoRA Adapters for Production

You do NOT run the LoRA adapter alongside the base model in production (it adds latency).
Before deploying, you mathematically merge the adapter into the base model.

Since $h = Wx + (BA)x$, and matrix multiplication is distributive:
$h = (W + BA)x$

You compute $W_{new} = W + BA$, save the new weights, and serve it exactly like a standard model using vLLM or Triton.

---

## 4. Production Architecture

### Automated LLMOps Fine-Tuning Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Enterprise MLOps Pipeline                         │
│                                                                        │
│  1. Data Prep (Airflow/Spark)                                          │
│     └── Clean JSONL dataset ──> Save to S3 (s3://data/train.jsonl)     │
│                                                                        │
│  2. Trigger Training (GitHub Actions / GitLab CI)                      │
│     └── POST /api/v1/jobs/finetune                                     │
│                                                                        │
│  3. Distributed Training Job (Kubernetes / Ray)                        │
│     ┌────────────────────────────────────────────────────────┐         │
│     │ Node: 1x p4d.24xlarge (8x A100 40GB)                   │         │
│     │ Framework: HuggingFace TRL (SFTTrainer) + DeepSpeed    │         │
│     │ Strategy: ZeRO-2 + LoRA (Rank 64)                      │         │
│     │ Tracking: MLflow / Weights & Biases                    │         │
│     └────────────────────────────────────────────────────────┘         │
│                                                                        │
│  4. Artifact Generation                                                │
│     ├── Base Model + LoRA Adapter (.safetensors)                       │
│     └── Merge step: create single production artifact                  │
│                                                                        │
│  5. Evaluation (LLM-as-a-Judge)                                        │
│     └── Run against held-out benchmark set                             │
│                                                                        │
│  6. Deployment (KServe / vLLM)                                         │
│     └── Deploy merged model to inference cluster                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Production Incident Scenarios

### Incident 1: "Loss Stays Flat / Model Does Not Learn"
**Symptoms:** During LoRA fine-tuning, the training loss curve is completely flat. Evaluation shows the model simply repeating base-model outputs.
**Root Cause:** The learning rate was set to `2e-5` (the standard for full fine-tuning). LoRA only trains a tiny fraction of parameters, so it requires a much higher learning rate.
**Fix:** Increase the learning rate for LoRA to `2e-4` or `3e-4`.

### Incident 2: "Catastrophic Forgetting after LoRA"
**Symptoms:** You fine-tuned a coding model to answer IT support tickets. It answers tickets well, but completely forgot how to write Python code.
**Root Cause:** The LoRA rank ($r$) was set too high (e.g., $r=256$) or `alpha` was too high, allowing the adapter to overpower the base model representations. Furthermore, the dataset lacked "replay" data.
**Fix:** 
1. Reduce Rank to $r=16$ or $r=32$.
2. Mix 10-20% of the original general training data (e.g., SlimPajama) into your fine-tuning dataset to anchor the model's general knowledge.

### Incident 3: "OOM on Long Context Fine-Tuning"
**Symptoms:** QLoRA training runs fine on 2,048 token sequences, but OOMs immediately when trying to fine-tune on 8,192 token sequences.
**Root Cause:** While QLoRA saves massive weight memory, **Activation Memory** scales quadratically with sequence length $O(N^2)$ due to attention.
**Fix:** Enable **Gradient Checkpointing** (Activation Checkpointing). This discards activations during the forward pass and recomputes them during the backward pass. It slows training by ~20% but reduces memory usage by ~70%.

---

## 6. Performance Optimization

### Distributed LoRA (FSDP + LoRA)

If your model is too large for 1 GPU even with QLoRA (e.g., Llama 405B), you must combine LoRA with FSDP (Fully Sharded Data Parallelism).

*Warning:* QLoRA (4-bit NF4) is historically incompatible with FSDP because FSDP requires parameters to be in floating-point formats to shard them. 
*Solution:* Use **HSDP** (Hybrid Sharded Data Parallel) or newer Answer.AI QLoRA+FSDP integrations, or stick to 16-bit LoRA combined with DeepSpeed ZeRO-3.

### Optimizing LoRA Hyperparameters

| Parameter | Recommended Value | Impact |
|---|---|---|
| **Rank (r)** | 8 to 64 | Higher rank = more capacity, but prone to overfitting. Start at 16. |
| **Alpha** | 2 × r | Scaling factor for the adapter. Usually set to double the rank. |
| **Target Modules** | `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Targeting all linear layers yields much better results than just targeting attention (`q`, `v`). |
| **Dropout** | 0.05 | Prevents overfitting on small datasets. |
| **Learning Rate** | `2e-4` | 10x higher than full fine-tuning. |
| **Batch Size** | 128 (global) | Use gradient accumulation to achieve this on small hardware. |

---

## 7. Hands-On: K8s Ray Training Job

```yaml
# Kubernetes RayJob for Distributed LoRA
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: lora-finetune
spec:
  entrypoint: python train_lora.py --model meta-llama/Llama-3-8B --data s3://data/train.jsonl
  rayClusterSpec:
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 2
      template:
        spec:
          containers:
          - name: ray-worker
            image: huggingface/transformers-pytorch-deepspeed-latest
            resources:
              limits:
                nvidia.com/gpu: 4  # Total 8 GPUs across 2 nodes
            volumeMounts:
            - mountPath: /dev/shm
              name: dshm
          volumes:
          - name: dshm
            emptyDir:
              medium: Memory
```

---

## Summary

```
Fine-Tuning Decision Tree:

Do you have $100k+ compute budget and millions of rows of data?
  ├── YES → Full Parameter Fine-Tuning (FSDP / Megatron)
  └── NO → Do you have < 24GB VRAM per GPU?
      ├── YES → QLoRA (4-bit base model + 16-bit adapter)
      └── NO → LoRA (16-bit base model + 16-bit adapter)

Production Workflow:
1. Train LoRA adapter.
2. Evaluate adapter.
3. Merge adapter into base weights (W_new = W_base + A*B).
4. Serve merged model using vLLM/Triton exactly like a standard model.
```

---
*Next: [11 — Embeddings, Vector Databases, Hybrid Search & RAG →](11_Embeddings_VectorDB_RAG_Deep_Dive.md)*
