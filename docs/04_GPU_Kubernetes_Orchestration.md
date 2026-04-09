# 04. GPU Kubernetes Orchestration (10/10 Enterprise Depth)

Kubernetes (K8s) relies on completely abstracting CPU and RAM via cgroups. GPUs, however, are massive hardware PCIe blobs that do not cleanly conform to the standard cgroup model. Proper orchestration demands strict Device Plugin management and topological awareness.

---

## 1. The NVIDIA GPU Operator

We do not install drivers using `apt-get` on production K8s nodes. The NVIDIA GPU Operator deploys everything as daemonsets. 

**Critical Core Components of the Operator:**
1.  `nvidia-driver-daemonset`: Compiles the driver against the current K8s node OS kernel version upon boot.
2.  `nvidia-container-toolkit`: Monkey-patches the container runtime (`containerd` or `docker`) to mount the `/dev/nvidia*` char devices natively inside containers.
3.  `nvidia-device-plugin`: Scans the PCI bus, detects GPUs, and advertises them to the Kubelet as `nvidia.com/gpu: 8`.

---

## 2. Multi-Instance GPU (MIG): Precise Hardware Partitioning

If a Data Scientist requests 1 GPU for an IDE session but only uses 5GB of vRAM, an entire 80GB A100 is wasted. Because Kubernetes fundamentally allocates GPUs as integers (1, 2, 3...), we use MIG to physically slice the hardware.

MIG completely isolates cache, memory, and compute. A crash in `Slice-1` cannot impact `Slice-2`.

### Configuring MIG in K8s
To enable MIG, you modify the ClusterPolicy or the `default-mig-parted-config` ConfigMap managed by the GPU Operator.

**Example MIG ConfigMap (Slicing an A100-80GB):**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: default-mig-parted-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-1g.10gb:
        - devices: all
          mig-enabled: true
          mig-devices:
            "1g.10gb": 7
      all-3g.40gb:
        - devices: all
          mig-enabled: true
          mig-devices:
            "3g.40gb": 2
```
If you apply the `all-3g.40gb` profile, the Device Plugin will advertise `nvidia.com/mig-3g.40gb: 2` to the cluster. A developer then requests exactly that in their manifest:

```yaml
resources:
  limits:
    nvidia.com/mig-3g.40gb: 1
```

---

## 3. Distributed Training via MPI Operator

Multi-node training requires thousands of GPUs to work together seamlessly. You cannot just deploy 10 Pods randomly. They need SSH keys exchanged, strict topology awareness, and a master orchestrator.

**Kubeflow's MPI Operator** solves this by treating a multi-node job as a single Custom Resource Definition (CRD).

**Real-world MPIJob Manifest:**
```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: llama-finetuning
spec:
  slotsPerWorker: 8 # Number of GPUs per Node
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - image: ds-team/llama-trainer:v1
            name: mpi-launcher
            command:
            - mpirun
            - -np 
            - "32" # 4 workers * 8 GPUs
            - -bind-to 
            - none
            - -map-by 
            - slot
            - python3 
            - /app/train.py
    Worker:
      replicas: 4
      template:
        spec:
          # Strict affinity to keep nodes on the same Top-of-Rack Switch
          affinity:
            podAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
              - labelSelector:
                  matchExpressions:
                  - key: app
                    operator: In
                    values: ["llama-finetuning"]
                topologyKey: topology.kubernetes.io/zone
          containers:
          - image: ds-team/llama-trainer:v1
            name: mpi-worker
            resources:
              limits:
                nvidia.com/gpu: 8 
                rdma/hca: 1 # Demanding SR-IOV injected RDMA interfaces
```

### Pod vs Node Affinity Focus
Notice the `topology.kubernetes.io/zone`. If the K8s scheduler places Worker 1 in Availability Zone A, and Worker 2 in Availability Zone B, the latency of passing gradients over cross-AZ fiber optic lines will destroy your TFLOPS. MLOps dictates enforcing co-location at the network switch level.
