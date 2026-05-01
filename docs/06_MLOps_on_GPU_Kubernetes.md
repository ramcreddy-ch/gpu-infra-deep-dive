# 06. MLOps on GPU Kubernetes: The Complete Operations Guide

Running ML workloads on Kubernetes is one thing. Running them on GPU-accelerated Kubernetes clusters in production — with real SLAs, real cost pressures, and real users — is an entirely different beast. This chapter bridges the gap between "I deployed a model" and "I operate an ML platform."

**Author:** [Ramchandra Chintala](https://github.com/ramcreddy-ch)

---

## The MLOps Lifecycle on K8s (How It Actually Works)

Here's the thing most tutorials skip: in a real production setup, your ML lifecycle on Kubernetes isn't a clean circle. It's messy, and every stage has GPU-specific landmines.

```
Data Lake ──► Feature Store ──► Training Job (GPU) ──► Model Registry
                                     │                       │
                                     ▼                       ▼
                              Experiment Tracking      Validation Gate
                              (MLflow on K8s)         (Automated Tests)
                                                            │
                                                            ▼
Monitoring ◄── Inference Service (GPU) ◄── Canary Deploy ◄── Approved
    │                  │
    ▼                  ▼
Drift Alert        Auto-scaling (GPU-aware HPA)
    │
    ▼
Retrain Trigger (back to Training Job)
```

### What makes GPU K8s different from regular K8s?

I've managed both, and here's what caught me off guard when I first started managing GPU clusters:

1. **GPUs are not fungible.** You can't just swap an A100 for a T4 mid-job. Your pod spec, your CUDA version, your model precision — everything is tied to the specific GPU generation.
2. **Scheduling is brutal.** GPU nodes are expensive. If your scheduler places a training job across availability zones, your NCCL all-reduce will crawl. Standard K8s affinity rules aren't enough.
3. **Failure modes are hardware-level.** CPUs rarely fail. GPUs fail all the time — ECC errors, thermal throttling, NVLink degradation. Your platform has to detect and drain bad nodes automatically.

---

## Setting Up a Production GPU Cluster (Step by Step)

### Step 1: Install the NVIDIA GPU Operator

Don't try to install drivers manually on every node. The GPU Operator handles drivers, container runtime hooks, device plugins, and monitoring — all as DaemonSets.

```bash
# Add the NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install the GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set migManager.enabled=true \
  --set mig.strategy=mixed
```

After this, run `kubectl get pods -n gpu-operator` and you should see:
- `nvidia-driver-daemonset-*` — Installs NVIDIA drivers on every GPU node
- `nvidia-container-toolkit-*` — Hooks into containerd so containers can access `/dev/nvidia*`
- `nvidia-device-plugin-*` — Advertises `nvidia.com/gpu` resources to the K8s scheduler
- `nvidia-dcgm-exporter-*` — Exports GPU metrics to Prometheus

### Step 2: Label and Taint Your GPU Node Pools

You absolutely do not want random pods landing on your $30k/month GPU nodes.

```bash
# Label nodes by GPU type (critical for scheduling)
kubectl label nodes gpu-node-01 nvidia.com/gpu.product=NVIDIA-A100-SXM4-80GB
kubectl label nodes gpu-node-01 workload-type=gpu-training

# Taint GPU nodes so only GPU workloads can schedule there
kubectl taint nodes gpu-node-01 nvidia.com/gpu=present:NoSchedule
```

Your training pods then need a matching toleration:
```yaml
tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
nodeSelector:
  workload-type: gpu-training
```

### Step 3: Configure DCGM Exporter + Prometheus

The DCGM Exporter exposes metrics that `nvidia-smi` can't. Here's a ServiceMonitor to scrape them:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: gpu-operator
spec:
  selector:
    matchLabels:
      app: nvidia-dcgm-exporter
  endpoints:
    - port: metrics
      interval: 15s
  namespaceSelector:
    matchNames:
      - gpu-operator
```

**Key metrics I watch daily:**
| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| `DCGM_FI_DEV_GPU_UTIL` | SM (Streaming Multiprocessor) activity | < 30% means GPU is starving |
| `DCGM_FI_DEV_FB_USED` | Framebuffer (vRAM) usage in MB | > 95% = OOM risk |
| `DCGM_FI_DEV_GPU_TEMP` | Core temperature in °C | > 83°C = throttling starts |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw in Watts | Spikes indicate workload bursts |
| `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | PCIe link retransmissions | > 0 = hardware degradation |
| `DCGM_FI_DEV_XID_ERRORS` | Hardware error codes | Any value = investigate |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory bandwidth utilization | > 90% = memory-bound workload |

---

## Running a Real Training Job on K8s

Here's how I actually submit a distributed LLaMA fine-tuning job. Not a toy example — this is what a real multi-node training manifest looks like.

```yaml
apiVersion: kubeflow.org/v2beta1
kind: PyTorchJob
metadata:
  name: llama-finetune-v2
  namespace: ml-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
            - name: pytorch
              image: registry.internal/ml-training:cuda12.2-pt2.1
              command: ["torchrun"]
              args:
                - "--nproc_per_node=8"
                - "--nnodes=4"
                - "--node_rank=$(INDEX)"
                - "--master_addr=$(MASTER_ADDR)"
                - "--master_port=29500"
                - "train.py"
                - "--model_name=meta-llama/Llama-2-7b"
                - "--batch_size=4"
                - "--gradient_accumulation_steps=8"
                - "--bf16"
                - "--deepspeed_config=ds_config.json"
              env:
                - name: NCCL_DEBUG
                  value: "WARN"
                - name: NCCL_IB_DISABLE
                  value: "0"
                - name: NCCL_SOCKET_IFNAME
                  value: "eth0"
              resources:
                limits:
                  nvidia.com/gpu: 8
                  memory: "256Gi"
                  cpu: "64"
              volumeMounts:
                - name: shm
                  mountPath: /dev/shm
                - name: dataset
                  mountPath: /data
          volumes:
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "64Gi"
            - name: dataset
              persistentVolumeClaim:
                claimName: training-dataset-pvc
          tolerations:
            - key: "nvidia.com/gpu"
              operator: "Exists"
              effect: "NoSchedule"
          affinity:
            podAntiAffinity:
              preferredDuringSchedulingIgnoredDuringExecution:
                - weight: 100
                  podAffinityTerm:
                    topologyKey: "kubernetes.io/hostname"
                    labelSelector:
                      matchLabels:
                        training.kubeflow.org/job-name: llama-finetune-v2
    Worker:
      replicas: 3
      # ... same spec as Master
```

**Things I learned the hard way:**
- `/dev/shm` must be a 64GB tmpfs mount. PyTorch's DataLoader uses shared memory for IPC between workers. The default 64MB will crash your job silently.
- `NCCL_IB_DISABLE=0` explicitly enables InfiniBand. On some cloud providers, NCCL defaults to TCP sockets which is 10x slower.
- `podAntiAffinity` on hostname ensures workers spread across physical nodes. Without this, K8s might pack all 4 workers on the same node if it has 32 GPUs.

---

## Model Serving on GPU K8s

### KServe with GPU: Production Config

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: fraud-detector-v3
  namespace: model-serving
  annotations:
    serving.kserve.io/autoscalerClass: "hpa"
    serving.kserve.io/targetUtilizationPercentage: "70"
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 20
    model:
      modelFormat:
        name: pytorch
      storageUri: "s3://models/fraud-detector/v3"
      resources:
        requests:
          nvidia.com/gpu: 1
          cpu: "4"
          memory: "16Gi"
        limits:
          nvidia.com/gpu: 1
          cpu: "8"
          memory: "32Gi"
    nodeSelector:
      workload-type: gpu-serving
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```

### GPU-Aware Autoscaling with KEDA

Standard K8s HPA doesn't understand GPU metrics. I use KEDA with Prometheus triggers to scale based on actual GPU utilization:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: fraud-detector-scaler
  namespace: model-serving
spec:
  scaleTargetRef:
    apiVersion: serving.kserve.io/v1beta1
    kind: InferenceService
    name: fraud-detector-v3
  minReplicaCount: 2
  maxReplicaCount: 20
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: inference_requests_per_second
        query: |
          sum(rate(inference_request_total{model="fraud-detector-v3"}[2m]))
        threshold: "100"
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        metricName: gpu_utilization
        query: |
          avg(DCGM_FI_DEV_GPU_UTIL{pod=~"fraud-detector.*"})
        threshold: "80"
```

---

## Cost Optimization: Because GPUs Are Expensive

This is the part that keeps my manager happy. Here's what I actually do to keep GPU costs under control:

### 1. Spot/Preemptible Instances for Training
Training jobs are fault-tolerant (we checkpoint every epoch). So I run them on Spot instances that are 60-70% cheaper.

```yaml
# EKS managed node group with Spot
nodeGroups:
  - name: gpu-training-spot
    instanceTypes: ["p4d.24xlarge", "p4de.24xlarge"]
    capacityType: SPOT
    labels:
      workload-type: gpu-training
      instance-lifecycle: spot
    taints:
      - key: nvidia.com/gpu
        value: "present"
        effect: NoSchedule
```

### 2. Time-Slicing for Development
Data Scientists experimenting in Jupyter don't need a full A100. I time-slice development GPUs:

```yaml
# GPU Operator time-slicing config
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4  # 1 physical GPU appears as 4 virtual GPUs
```

### 3. Cluster Autoscaler with Scale-to-Zero
GPU nodes sitting idle burn money. I configure scale-to-zero for non-critical workloads:

```yaml
# Cluster Autoscaler annotation on the node group
cluster-autoscaler.kubernetes.io/scale-down-enabled: "true"
cluster-autoscaler.kubernetes.io/scale-down-delay-after-add: "10m"
cluster-autoscaler.kubernetes.io/scale-down-unneeded-time: "5m"
```

---

*Next: [07. LLMOps: Serving Large Language Models at Scale](./07_LLMOps_Serving_LLMs_at_Scale.md)*
