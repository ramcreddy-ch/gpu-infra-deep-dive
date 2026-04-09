# 04. GPU Orchestration in Kubernetes

Running GPUs on bare metal is easy. Running thousands of them efficiently inside Kubernetes is where Platform Engineering becomes an art form.

## 1. The NVIDIA GPU Operator
Never install drivers manually on K8s nodes. Use the **NVIDIA GPU Operator**.
The Operator relies on the Operator Framework to automate the lifecycle of all NVIDIA software components needed to provision GPUs:
*   GPU Drivers (via DaemonSet)
*   NVIDIA Container Toolkit (adjusts `runc` to `nvidia-container-runtime`)
*   Kubernetes Device Plugin
*   DCGM Exporter (for Prometheus metrics)
*   MIG Manager

### The Device Plugin
The workhorse. It listens to the kubelet and exposes GPUs as allocatable resources (`nvidia.com/gpu: 1`).
**Note:** Standard Kubernetes treats a GPU as an *integer*. You request `1` GPU, you get the whole 80GB card. Kubernetes does not natively understand GPU memory sharing.

## 2. Multi-Tenancy: Slicing the GPU

If an intern requests a GPU for a Jupyter Notebook that uses 4GB of vRAM, giving them an entire 80GB H100 is throwing away thousands of dollars. We must share GPUs.

### Option A: Time-Slicing (Software Level)
You configure the Device Plugin to lie to Kubernetes and say the node has `10` GPUs when it physically has `1`. 
*   **Pros:** Easy to set up.
*   **Cons:** No hardware isolation. Workload A can still OOM and crash Workload B because they share the same physical memory space. Context switching causes latency.

### Option B: MIG (Multi-Instance GPU - Hardware Level)
Available on Ampere (A100) and Hopper (H100) architecture. Physically partitions the GPU into up to 7 isolated instances at the hardware level. Each gets its own L2 cache, memory bandwidth, and compute.
*   **Pros:** Strict isolation. Workload A cannot impact Workload B. Zero cross-talk.
*   **Cons:** Static allocation. You must slice the GPU (e.g., into two 40GB slices or seven 10GB slices) in advance. It requires node reboots or draining to reconfigure topologies. 

## 3. Advanced Scheduling in K8s

### Taints and Tolerations
GPUs are expensive. You do not want a typical web-server pod scheduling onto a GPU node.
**Rule:** Always Taint your GPU nodes.
`kubectl taint nodes gpu-node-01 nvidia.com/gpu=true:NoSchedule`

Data Scientists must explicitly add the Toleration in their Pod Spec to access the hardware:
```yaml
tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

### Node Affinity and Topology
For multi-node distributed jobs (MPI/NCCL), physics matters. If you schedule half the pods on Rack A and half on Rack Z, network hops will destroy performance.
Use **Pod Affinity** and **Topology Spread Constraints** to pack training pods onto the exact same Top-of-Rack switch to minimize RDMA latency. 
