<div align="center">
  <h1>🚀 GPU Infrastructure Deep Dive</h1>
  <p><b>The Ultimate, Battle-Tested Guide to Managing, Optimizing, and Troubleshooting AI/ML GPU Workloads at Scale.</b></p>
  <p><i>Authored by Ramchandra Chintala | </i></p> 
</div>

---

## 📌 Introduction
Welcome to the definitive guide on **GPU Infrastructure for AI & Deep Learning**. As AI workloads (LLMs, GenAI models, Diffusion) scale to billions of parameters, treating the GPU simply as a "faster CPU" is a critical anti-pattern. 

Drawing from years of hands-on experience designing distributed training pipelines, battling multi-node NCCL timeouts, and squeezing every drop of FLOPS from A100/H100 clusters, I created this repository as a **one-stop resource**. Whether you are transitioning from traditional DevOps to MLOps, or you are a seasoned Platform Engineer facing "CUDA Out Of Memory" panics in production, this repo is for you.

## 🎯 What to Expect?
- **Raw, Unfiltered Realities**: No hello-world tutorials here. This repo tackles real-time production issues like Thermal Throttling, High NVLink Latency, and GPU Zombie Processes.
- **From Zero to Architect**: We start with how a GPU actually works under the hood and progress toward orchestrating Multi-Instance GPUs (MIG) over Kubernetes multi-node fabrics.
- **Scripted Tooling**: Direct access to production-ready scripts for monitoring and debugging bare-metal and containerized GPU states.

## 📚 Table of Contents

### Core Documentation
1. [**01. GPU Architecture Basics: Under the Silicon Hood**](./docs/01_GPU_Architecture_Basics.md)
   - CUDA Cores vs Tensor Cores, HBM Memory Bandwidth, PCIe/NVLink/NVSwitch Topologies.
2. [**02. GPU AI Workloads & Utilization**](./docs/02_GPU_AI_Workloads_and_Utilization.md)
   - Dissecting LLM Training vs Inference, Memory Fragmentation, Precision (FP32/FP16/BF16/INT8), and vRAM math.
3. [**03. Real-Time Issues & Troubleshooting (The War Room)**](./docs/03_Real_Time_Issues_Troubleshooting.md)
   - **Real fixes** for 100% Volatile GPU-Util but 0% SM-Util, Zombie memory leaks, PCIe uncorrectable AER errors, Thermal degradation.
4. [**04. GPU Orchestration in Kubernetes**](./docs/04_GPU_Kubernetes_Orchestration.md)
   - Advanced scheduling: NVIDIA GPU Operator, Time-Slicing vs MIG, GPU Taints, Node Affinity, and RDMA setups.
5. [**05. Advanced Optimizations**](./docs/05_Advanced_Optimizations.md)
   - Operator fusion, using FlashAttention, DeepSpeed Zero strategies, TensorRT compilation, and vLLM continuous batching.

### Automation & Tooling
- `scripts/gpu_health_metrics.py` - Custom Prometheus exporter-like script leveraging `pynvml` to catch throttling before your models crash.
- `scripts/zombie_killer.sh` - Safe reaping of orphaned processes holding precious vRAM hostage.

---


<div align="center">
  <i>"The difference between a failing AI start-up and a profitable one often lies in how effectively their clusters utilize hardware." <br>— Ramchandra Chintala</i>
</div>

Hi, I am **Ramchandra Chintala**, a Senior Platform, Streaming, and MLOps Engineer. I specialize in architecting hyper-scale AI platforms. This repository is derived directly from the trenches of managing large-scale, high-density AI clusters. 

While I may not hold every official paper certificate out there, **by reading, understanding, and applying the playbooks in this repository, you will operate at the level of a certified Advanced GPU and AI Infrastructure Expert.**


---
**License:** MIT License. Feel free to use these playbooks to stabilize your own clusters!
