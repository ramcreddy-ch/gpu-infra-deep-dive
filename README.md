<div align="center">
  <h1>🚀 GPU Infrastructure Deep Dive</h1>
  <p><b>The Ultimate, Battle-Tested Guide to Managing, Optimizing, and Troubleshooting AI/ML GPU Workloads at Scale.</b></p>
  <p><i>Authored by <a href="https://github.com/ramcreddy-ch">Ramchandra Chintala</a> — Senior MLOps & Platform Engineer</i></p>
</div>

---

## 📌 Introduction

Welcome to the definitive guide on **GPU Infrastructure for AI & Deep Learning**. As AI workloads (LLMs, GenAI models, Diffusion) scale to billions of parameters, treating the GPU simply as a "faster CPU" is a critical anti-pattern.

Drawing from years of hands-on experience designing distributed training pipelines, battling multi-node NCCL timeouts, and squeezing every drop of FLOPS from A100/H100 clusters, I created this repository as a **one-stop resource**. Whether you are transitioning from traditional DevOps to MLOps, or you are a seasoned Platform Engineer facing "CUDA Out Of Memory" panics in production, this repo is for you.

## 🎯 What to Expect?
- **Raw, Unfiltered Realities**: No hello-world tutorials here. This repo tackles real-time production issues like Thermal Throttling, High NVLink Latency, and GPU Zombie Processes.
- **From Zero to Architect**: We start with how a GPU actually works under the hood and progress toward orchestrating Multi-Instance GPUs (MIG) over Kubernetes multi-node fabrics.
- **Scripted Tooling**: Direct access to production-ready scripts for monitoring and debugging bare-metal and containerized GPU states.
- **MLOps + LLMOps**: Complete guides on running ML training jobs, serving LLMs with vLLM/Triton, quantization, and prompt routing — all on GPU Kubernetes clusters.
- **Real War Stories**: Actual production incidents I've dealt with, including thermal throttling disguised as 100% utilization, $12k Jupyter notebooks, and NCCL hangs caused by a single degrading NVLink.

---

## 📚 Table of Contents

### Part 1: GPU Fundamentals
1. [**01. GPU Architecture Basics: Under the Silicon Hood**](./docs/01_GPU_Architecture_Basics.md)
   - CUDA Cores vs Tensor Cores, HBM Memory Bandwidth, PCIe/NVLink/NVSwitch Topologies.
2. [**02. GPU AI Workloads & Utilization**](./docs/02_GPU_AI_Workloads_and_Utilization.md)
   - Dissecting LLM Training vs Inference, Memory Fragmentation, Precision (FP32/FP16/BF16/INT8), and vRAM math.
3. [**03. Real-Time Issues & Troubleshooting (The War Room)**](./docs/03_Real_Time_Issues_Troubleshooting.md)
   - **Real fixes** for 100% Volatile GPU-Util but 0% SM-Util, Zombie memory leaks, PCIe uncorrectable AER errors, Thermal degradation.

### Part 2: GPU on Kubernetes
4. [**04. GPU Orchestration in Kubernetes**](./docs/04_GPU_Kubernetes_Orchestration.md)
   - Advanced scheduling: NVIDIA GPU Operator, Time-Slicing vs MIG, GPU Taints, Node Affinity, and RDMA setups.
5. [**05. Advanced Optimizations**](./docs/05_Advanced_Optimizations.md)
   - Operator fusion, using FlashAttention, DeepSpeed ZeRO strategies, TensorRT compilation, and vLLM continuous batching.
6. [**06. MLOps on GPU Kubernetes**](./docs/06_MLOps_on_GPU_Kubernetes.md) ⭐ **NEW**
   - Setting up production GPU clusters, DCGM monitoring, running real distributed training jobs (PyTorchJob), KServe model serving with GPU-aware KEDA autoscaling, and cost optimization (Spot instances, time-slicing, scale-to-zero).

### Part 3: LLMOps & Production AI
7. [**07. LLMOps: Serving Large Language Models at Scale**](./docs/07_LLMOps_Serving_LLMs_at_Scale.md) ⭐ **NEW**
   - vLLM deployment on K8s, NVIDIA Triton, quantization comparison (GPTQ/AWQ/GGUF/FP8), prompt routing gateway architecture, and real LLMOps production issues.
8. [**08. Day-to-Day GPU Operations & War Stories**](./docs/08_Day_to_Day_GPU_Operations.md) ⭐ **NEW**
   - Real production war stories: thermal throttling disguised as 100% utilization, zombie process 503 avalanches, NCCL hangs from NVLink degradation, $12k Jupyter notebooks, and BF16/T4 compatibility failures. Includes daily kubectl checklists and a pre-deployment checklist.

### Part 4: Security & Observability
9. [**09. GPU Security, Compliance, and Multi-Tenancy**](./docs/09_GPU_Security_Compliance.md) ⭐ **NEW**
   - MIG isolation for multi-tenancy, model weight encryption, Pod Security Standards for GPU containers, RBAC for GPU access, Falco rules for unauthorized GPU access, and SOC 2/HIPAA compliance.
10. [**10. GPU Monitoring Deep Dive & Prometheus Metrics**](./docs/10_GPU_Monitoring_Prometheus.md) ⭐ **NEW**
    - The exact DCGM metrics that matter (filtered from 100+), tiered Prometheus alert rules, Grafana dashboard structure, custom PyTorch GPU metrics exporter, and a one-command diagnostic snapshot script.

---

### Automation & Tooling
| Script | Purpose |
|--------|---------|
| `scripts/gpu_health_metrics.py` | Custom Prometheus exporter leveraging `pynvml` to catch throttling before your models crash |
| `scripts/zombie_killer.sh` | Safe reaping of orphaned processes holding precious vRAM hostage |
| `scripts/gpu_snapshot.sh` ⭐ **NEW** | One-command GPU node diagnostics: temps, ECC errors, PCIe status, NVLink health, zombie PIDs, and kernel errors |
| `scripts/gpu_idle_detector.py` ⭐ **NEW** | Queries Prometheus for idle GPU pods, estimates dollar waste, and reports which pods to reclaim |

---

## 🏗️ Architecture Coverage

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        GPU INFRASTRUCTURE LAYERS                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Application Layer                                                  │     │
│  │  MLflow • KServe • vLLM • Triton • Feast • Airflow                 │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Orchestration Layer                                                │     │
│  │  K8s GPU Operator • MIG Manager • KEDA • Device Plugin             │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Runtime Layer                                                      │     │
│  │  CUDA 12.x • cuDNN • NCCL • TensorRT • DeepSpeed • FlashAttention │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Hardware Layer                                                     │     │
│  │  A100/H100 • NVLink/NVSwitch • PCIe Gen5 • InfiniBand • HBM3      │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Observability Layer                                                │     │
│  │  DCGM Exporter • Prometheus • Grafana • Custom Metrics • Alerting  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/ramcreddy-ch/gpu-infra-deep-dive.git
cd gpu-infra-deep-dive

# Run GPU diagnostics on any GPU node
bash scripts/gpu_snapshot.sh

# Check for idle GPU pods wasting money
python scripts/gpu_idle_detector.py --prometheus-url http://prometheus:9090

# Scan GPU health and export metrics
python scripts/gpu_health_metrics.py

# Kill zombie processes holding GPU vRAM
bash scripts/zombie_killer.sh
```

---

**Built with ❤️ by [Ramchandra Chintala](https://github.com/ramcreddy-ch)**

*Senior MLOps & Platform Engineer — GPU Infrastructure, Kubernetes, and AI/ML at Scale*
