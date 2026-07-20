# Kubernetes GPU Orchestration, NVIDIA GPU Operator, MIG, Device Plugin & Scheduling

> *Kubernetes was designed to orchestrate stateless CPU workloads. Retrofitting it to orchestrate highly stateful, expensive, and topology-sensitive GPUs is one of the hardest challenges in Platform Engineering. This chapter explains how to build a production AI Kubernetes cluster.*

---

## 1. What Problem Does This Solve?

### The "Bare Metal" AI Anti-Pattern

In 2018, data scientists SSH'd directly into bare-metal GPU servers. 
- **Dependency Hell:** Alice needs CUDA 11.8 for PyTorch. Bob needs CUDA 12.1 for TensorRT. They break each other's environments daily.
- **Resource Hoarding:** Charlie runs a notebook on a $250k DGX node, uses 5% of one GPU, and leaves for the weekend. No one else can use it.
- **No Fault Tolerance:** If node 3 dies, training stops until an admin manually moves the scripts.

### Why Kubernetes for AI?

Kubernetes solves this through **containerization and scheduling**:
1. **Isolation:** Alice and Bob package their specific CUDA versions in Docker images.
2. **Bin-packing:** The scheduler packs multiple workloads onto a single node to maximize utilization.
3. **Resilience:** If a Spot GPU node is preempted, K8s reschedules the training pod automatically.

**The Challenge:** Kubernetes natively only understands `cpu` and `memory`. It does not know what a GPU is. We have to teach it.

---

## 2. Internal Architecture

### The NVIDIA K8s Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Kubernetes Worker Node (GPU)                         │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Kubelet (Node Agent)                          │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
│                                    │ gRPC (Device Plugin API)           │
│  ┌─────────────────────────────────▼─────────────────────────────────┐  │
│  │                NVIDIA Device Plugin (DaemonSet)                   │  │
│  │  • Scans hardware for GPUs                                        │  │
│  │  • Registers 'nvidia.com/gpu' resource with Kubelet               │  │
│  │  • Injects /dev/nvidiaX into Pod cgroups                          │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
│                                    │ Container creation request         │
│  ┌─────────────────────────────────▼─────────────────────────────────┐  │
│  │           NVIDIA Container Toolkit (nvidia-ctk)                   │  │
│  │  • Intercepts runc / containerd                                   │  │
│  │  • Mounts NVIDIA driver libraries (.so) into container            │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
│                                    │                                    │
│  ┌─────────────────────────────────▼─────────────────────────────────┐  │
│  │                Linux Kernel & NVIDIA Driver                       │  │
│  │  /dev/nvidia0   /dev/nvidia1   /dev/nvidiactl   /dev/nvidia-uvm   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### The NVIDIA GPU Operator

Deploying the stack manually is a nightmare (drivers must match kernels, toolkits must match container runtimes). 
The **NVIDIA GPU Operator** is a single Helm chart that deploys everything via Custom Resource Definitions (CRDs).

**What the Operator deploys (DaemonSets):**
1. **NVIDIA Driver Container:** Compiles and loads the `.ko` kernel modules dynamically.
2. **NVIDIA Container Toolkit:** Configures `containerd` to use the NVIDIA runtime.
3. **NVIDIA Device Plugin:** Exposes `nvidia.com/gpu` to K8s.
4. **DCGM Exporter:** Exposes GPU Prometheus metrics.
5. **Node Feature Discovery (NFD):** Labels the node with hardware specs.
6. **MIG Manager:** Automates partitioning of A100/H100 GPUs.

---

## 3. Deep Internal Working

### How a Pod Gets a GPU (Step-by-Step)

When you submit this pod:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
spec:
  containers:
  - name: pytorch
    image: pytorch/pytorch:latest
    resources:
      limits:
        nvidia.com/gpu: 2
```

**The Execution Flow:**
1. **API Server:** Accepts the Pod.
2. **Scheduler:** Looks for a node with `Capacity: nvidia.com/gpu: >= 2`. Assigns Pod to Node X.
3. **Kubelet on Node X:** Sees the new Pod. It calls the Device Plugin API: `Allocate()`.
4. **Device Plugin:** Selects two specific physical GPUs (e.g., GPU 1 and GPU 3). Returns their device paths and environment variables (e.g., `NVIDIA_VISIBLE_DEVICES=1,3`).
5. **Containerd:** Starts creating the container via `runc`.
6. **NVIDIA Container Toolkit:** A pre-start hook intercepts `runc`. It binds-mounts the driver libraries from the host into the container (e.g., `libcuda.so`, `libnvcuvid.so`).
7. **cgroups:** The Linux kernel `devices` cgroup is updated to allow the container process to read/write ONLY to `/dev/nvidia1` and `/dev/nvidia3`.
8. **Process Starts:** When PyTorch calls `cudaGetDeviceCount()`, it sees exactly 2 GPUs.

### Multi-Instance GPU (MIG) vs. Time-Slicing

If you request `nvidia.com/gpu: 1`, you get an entire physical GPU. What if your model only needs 4GB of VRAM? You waste the other 76GB on an H100.

**Approach 1: Time-Slicing (Concurrency without Isolation)**
- You configure the Device Plugin to advertise `nvidia.com/gpu: 10` for a single physical GPU.
- K8s schedules 10 pods to the same GPU.
- **How it works internally:** All 10 processes share the same SMs and VRAM. The GPU firmware context-switches between them.
- **The Problem:** No memory isolation. If Pod A allocates 80GB, Pods B-J crash with CUDA OOM. No fault isolation.

**Approach 2: MIG (Hardware Isolation)**
- Supported on Ampere (A100) and Hopper (H100).
- Partitions a physical GPU into up to 7 distinct silicon slices.
- Each slice gets dedicated SMs, dedicated L2 cache, and dedicated VRAM.
- **How it works in K8s:** The GPU Operator detects MIG slices and advertises them as distinct resources: `nvidia.com/mig-1g.10gb: 7`.
- **The Result:** Perfect isolation. Pod A cannot crash Pod B. It behaves exactly like 7 physical mini-GPUs.

---

## 4. Production Architecture

### Enterprise AI Kubernetes Cluster Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster (EKS/GKE)                    │
│                                                                        │
│  ┌─────────────────────────┐  ┌────────────────────────────────────┐   │
│  │ Node Pool: cpu-general  │  │ Node Pool: gpu-inference-a10g      │   │
│  │ (M5 / N2 instances)     │  │ (G5 instances, Taints: gpu=true)   │   │
│  │ • API Gateways          │  │ • vLLM Pods                        │   │
│  │ • Ray Head Nodes        │  │ • Triton Pods                      │   │
│  │ • KServe Controllers    │  │ • Embedding Models                 │   │
│  └─────────────────────────┘  └────────────────────────────────────┘   │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Node Pool: gpu-training-h100 (P5 instances)                     │   │
│  │ Taints: nvidia.com/gpu=true:NoSchedule                          │   │
│  │ Topology: Requires strict placement in same placement group     │   │
│  │ • PyTorch Distributed Training (Volcano / Ray Train)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

**Key Architectural Design: Node Feature Discovery (NFD) Labels**
Never schedule a Pod to a random GPU node. Use NodeSelectors based on NFD labels.
```yaml
nodeSelector:
  nvidia.com/gpu.family: hopper  # Only run on H100s
  nvidia.com/gpu.memory: "81920" # Only run on 80GB cards
```

---

## 5. Production Incident Scenarios

### Incident 1: "The CUDA Version Mismatch Panic"
**Symptoms:** Developers deploy a pod. PyTorch throws `RuntimeError: cuda runtime error (35) : CUDA driver version is insufficient for CUDA runtime version`.
**Root Cause:** The host NVIDIA driver is version 525 (supports up to CUDA 12.0). The container image uses PyTorch compiled with CUDA 12.1.
**Fix:** The NVIDIA driver must ALWAYS be greater than or equal to the CUDA Toolkit version inside the container. Upgrade the driver via the GPU Operator Helm values.
**Staff Engineer Prevention:** Implement Kyverno policies in K8s to block container images that use CUDA versions higher than the cluster's driver supports.

### Incident 2: "Device Plugin CrashLoop — GPU Dropped off PCIe Bus"
**Symptoms:** Nodes show `nvidia.com/gpu: 0` capacity, but they are GPU instances. `nvidia-device-plugin` daemonset pods are CrashLoopBackOff.
**Logs:** `Failed to initialize NVML: Driver Not Loaded`
**Root Cause:** A hardware fault (often a thermal event or power spike) caused the GPU to "fall off" the PCIe bus. The OS no longer sees the PCI device (check `lspci | grep NVIDIA`).
**Fix:** Hard reboot the node. If it happens again, the motherboard PCIe slot or GPU is dead. Drain the node and RMA the hardware.

### Incident 3: "DDP Training Hangs Across Nodes"
**Symptoms:** 2-node PyTorch training job starts, but hangs before the first step.
**Root Cause:** The K8s CNI (Container Network Interface) is blocking the ephemeral ports PyTorch uses for NCCL communication, or the Pods do not have `hostNetwork: true` required for high-speed RDMA routing.
**Fix:** For multi-node training, you generally need `hostNetwork: true` or a specialized CNI (like Multus) to bypass the K8s overlay network and allow direct RDMA/InfiniBand communication between pods.

---

## 6. Performance Optimization

### Overcoming the Docker/K8s IPC Overhead

When running multi-process workloads (like PyTorch DataLoaders with `num_workers > 0`), PyTorch uses Linux shared memory (`/dev/shm`) to pass tensors between processes without copying them.

**The K8s Default Problem:** K8s sets `/dev/shm` to 64MB by default. A single batch of images will immediately crash the pod with `Bus error`.

**The Optimization (Mandatory for AI):**
Always mount an in-memory `emptyDir` to `/dev/shm`.
```yaml
volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: 32Gi  # Or roughly 50% of node RAM
```

### CPU Affinity and NUMA in K8s

For max performance on multi-GPU nodes (like 8x H100), the CPU process feeding GPU 0 MUST run on the CPU socket physically attached to GPU 0's PCIe root complex.

**K8s Configuration:**
Use the Kubernetes **Topology Manager** and **CPU Manager**.
1. Enable `cpuManagerPolicy: static` on the Kubelet.
2. Enable `topologyManagerPolicy: single-numa-node`.
3. Set CPU requests to integer values (e.g., `cpu: 16`).
This forces K8s to pin the pod's CPU threads to the exact same NUMA node as the allocated GPU.

---

## Summary

```
K8s GPU Troubleshooting Cheat Sheet:
1. Is the node labeled? 
   kubectl get nodes -l nvidia.com/gpu.present=true
2. Is the capacity registered? 
   kubectl get node <node-name> -o yaml | grep nvidia.com/gpu
3. Are the daemonsets running? 
   kubectl get pods -n gpu-operator
4. Check the device plugin logs: 
   kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

---
*Next: [08 — High-Speed Networking (NCCL, InfiniBand, RDMA) →](08_High_Speed_Networking_Deep_Dive.md)*
