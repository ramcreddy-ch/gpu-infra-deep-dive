# 08. Day-to-Day GPU Operations & War Stories

This isn't a tutorial. This is a collection of things that actually happened to me (and my team) while running GPU clusters in production. Every section here is a real scenario, a real fix, and a real lesson learned. If you're interviewing for a Senior MLOps/Platform Engineer role, this is the stuff they'll ask you about.

**Author:** [Ramchandra Chintala](https://github.com/ramcreddy-ch)

---

## What My Morning Looks Like

I'll be honest — most days are uneventful if the platform is set up right. But my morning routine hasn't changed in years:

1. **Check Grafana GPU dashboard.** I look at overnight GPU utilization across all clusters. If a training job was scheduled at 2 AM and the GPU util shows 0% from 2:15 AM onwards, something crashed silently.
2. **Review DCGM alerts.** Any Xid errors? Any thermal throttling events? Any PCIe replay counter spikes? These are early warnings that a GPU is about to die.
3. **Check Kubernetes events.** `kubectl get events --field-selector reason=FailedScheduling -n ml-training` tells me if any job couldn't find a GPU node.
4. **Glance at costs.** I have a custom Kubecost dashboard that shows GPU-hour spend per team. If the ML Research team burned $5k overnight on Spot instances for a hyperparameter sweep, that's fine. If the same cost came from a stuck debug pod, that's a problem.

---

## War Story #1: The Training Job That Wouldn't Die

**What happened:** A PyTorchJob was supposed to run for 8 hours. After 12 hours, it was still "Running" but the training loss hadn't changed in 4 hours. GPU utilization showed 100%.

**The investigation:**
- `kubectl logs` showed the last log line was a gradient sync step. No error.
- `nvidia-smi` inside the pod showed 100% GPU-Util.
- But `DCGM_FI_DEV_SM_CLOCK` showed the SM clock was running at 210 MHz instead of the normal 1410 MHz.

**Root cause:** Thermal throttling. The GPU temperature had hit 90°C because the data center cooling system partially failed overnight. The GPU didn't crash — it just slowed itself down to 15% speed to avoid damage. From the outside, it looked "busy."

**The fix:**
```bash
# Check current throttling reasons
nvidia-smi -q -d CLOCK | grep -A5 "Clocks Throttle Reasons"

# What you want to see:
#   HW Slowdown    : Not Active
#   SW Thermal Slowdown : Not Active

# What I saw:
#   HW Thermal Slowdown : Active
```

I immediately drained the node, filed an RMA with the hosting provider, and added a Prometheus alert:
```yaml
- alert: GPUThermalThrottling
  expr: DCGM_FI_DEV_GPU_TEMP > 83
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "GPU {{ $labels.gpu }} on {{ $labels.node }} is thermally throttling at {{ $value }}°C"
```

**Lesson learned:** 100% GPU utilization does NOT mean 100% performance. Always cross-reference with SM clock speed and power draw.

---

## War Story #2: The 503 Avalanche

**What happened:** Our fraud detection inference endpoint (KServe + GPU) suddenly started returning 503s at 9 AM on a Monday. Traffic hadn't increased.

**The investigation:**
- The KServe pod was healthy, no restarts.
- `nvidia-smi` showed only 2 GB of vRAM used (out of 80 GB).
- Pod logs showed: `torch.cuda.OutOfMemoryError: CUDA out of memory`

Wait — 2 GB used but OOM? That makes no sense.

**Root cause:** A zombie process from a crashed training job on the same GPU node was holding 78 GB of vRAM. `nvidia-smi` only shows processes visible in the current container's PID namespace. The zombie was in a different (dead) container's namespace.

**The fix:**
```bash
# On the host (not inside the container):
fuser -v /dev/nvidia0
# Output showed PID 42531 holding the device

# Check what it was
ps aux | grep 42531
# result: [python] <defunct>   ← zombie

kill -9 42531
# Memory freed instantly
```

**Lesson learned:** Always run `fuser -v /dev/nvidia*` on the host, not inside the container. Container-level `nvidia-smi` lies about who's using the GPU.

---

## War Story #3: NCCL All-Reduce Hanging on Node 3

**What happened:** A 4-node distributed training job (32 GPUs total) would hang after exactly 1 epoch. Nodes 1, 2, and 4 were waiting. Node 3 was unresponsive.

**The investigation:**
With `NCCL_DEBUG=INFO`, the logs showed:
```
Node 0: NCCL INFO AllReduce: opCount 847, sendbuff 0x7f... OK
Node 1: NCCL INFO AllReduce: opCount 847, sendbuff 0x7f... OK
Node 2: NCCL INFO AllReduce: opCount 847, sendbuff 0x7f... OK
Node 3: [no output after opCount 846]
```

Node 3 fell behind by exactly 1 operation.

**Root cause:** Node 3 had an NVLink that was degrading. It was still functional but its bandwidth had dropped from 600 GB/s to 50 GB/s. The data transfer for the all-reduce was taking 12x longer on Node 3, causing all other nodes to timeout (default NCCL timeout is 30 minutes, which it was hitting).

**The fix:**
```bash
# Check NVLink status on the node
nvidia-smi nvlink --status
# One link showed: "Link 2: <inactive>" ← physical damage

# Immediate fix: drain the node
kubectl drain gpu-node-03 --ignore-daemonsets --delete-emptydir-data
kubectl taint nodes gpu-node-03 hardware-issue=nvlink:NoSchedule
```

**Lesson learned:** NCCL timeouts are almost never a software problem. They're a topology problem. Either a network link is bad, a GPU is degraded, or the K8s scheduler placed your pods across AZs.

---

## War Story #4: The $12,000 Jupyter Notebook

**What happened:** My Kubecost dashboard showed a single user had consumed $12,000 in GPU-hours in one month. They had a Jupyter notebook on an A100 node that was running 24/7.

**The investigation:** I looked at the GPU utilization for their pod — it was 0% for 23 hours a day. They used the A100 for maybe 45 minutes of actual compute, then left the notebook open.

**The fix:** Implemented idle GPU detection and automatic pod eviction:

```yaml
# CronJob that checks GPU utilization every 30 minutes
apiVersion: batch/v1
kind: CronJob
metadata:
  name: gpu-idle-detector
  namespace: ml-platform
spec:
  schedule: "*/30 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: idle-checker
              image: bitnami/kubectl:latest
              command:
                - /bin/bash
                - -c
                - |
                  # Find pods with GPU requests that have <5% utilization for >2h
                  # In production: query Prometheus for DCGM_FI_DEV_GPU_UTIL
                  echo "Checking for idle GPU pods..."
                  # kubectl delete pod $IDLE_POD -n $NS --grace-period=300
          restartPolicy: OnFailure
```

Also set resource quotas per namespace:
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-research
spec:
  hard:
    requests.nvidia.com/gpu: "4"  # Max 4 GPUs per team
    limits.nvidia.com/gpu: "4"
```

**Lesson learned:** If you don't set GPU quotas and idle detection from day one, you WILL get a surprise bill.

---

## War Story #5: The Model That Worked on A100 but Failed on T4

**What happened:** A Data Scientist trained a model on an A100 (80GB, BF16 support) and shipped it to a T4 serving node (16GB, no BF16). The model loaded but produced garbage predictions.

**Root cause:** The model used BF16 (bfloat16) precision during training. T4 GPUs don't support BF16 natively — they silently cast to FP32, which changes the numerical behavior of some operations and blows past the 16GB vRAM limit.

**The fix:**
1. Added a validation step in CI/CD that checks model compatibility with the target GPU architecture.
2. Enforced explicit `nodeSelector` matching in all serving manifests:

```yaml
nodeSelector:
  nvidia.com/gpu.product: "NVIDIA-A10G"  # or T4, A100, etc.
  nvidia.com/gpu.compute.major: "8"       # Ampere = 8, Turing = 7
```

3. Added a pre-deployment model validation script:
```python
import torch

def validate_model_for_gpu(model_path, target_gpu="T4"):
    model = torch.load(model_path)
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16 and target_gpu in ["T4", "V100"]:
            raise ValueError(
                f"Parameter '{name}' uses BF16 which is not supported on {target_gpu}. "
                f"Convert to FP16 first: model.half()"
            )
```

**Lesson learned:** GPU generations are not backward-compatible for precision formats. Always validate model precision against the serving GPU before deploying.

---

## Quick Reference: kubectl Commands I Use Daily

```bash
# Check which nodes have GPUs and how many are allocated
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# Find pods requesting GPUs
kubectl get pods --all-namespaces -o json | \
  jq '.items[] | select(.spec.containers[].resources.limits["nvidia.com/gpu"] != null) | .metadata.name'

# Check GPU operator health
kubectl get pods -n gpu-operator

# View GPU metrics from DCGM
kubectl exec -it $(kubectl get pod -n gpu-operator -l app=nvidia-dcgm-exporter -o name | head -1) \
  -n gpu-operator -- dcgm-exporter -f /etc/dcgm-exporter/dcp-metrics-included.csv

# Emergency: drain a bad GPU node
kubectl drain gpu-node-03 --ignore-daemonsets --delete-emptydir-data --force
kubectl taint nodes gpu-node-03 hardware-issue=true:NoSchedule

# Check why a GPU pod is pending
kubectl describe pod stuck-training-pod -n ml-training | grep -A10 Events
```

---

## The Checklist I Run Before Every Major GPU Deployment

- [ ] GPU Operator is healthy (`kubectl get pods -n gpu-operator`)
- [ ] DCGM Exporter is scraping metrics into Prometheus
- [ ] GPU node taints and labels are correct
- [ ] `/dev/shm` is mounted as tmpfs with sufficient size
- [ ] NCCL environment variables are set for multi-node jobs
- [ ] Model precision matches the serving GPU architecture
- [ ] Resource quotas are set per team/namespace
- [ ] Idle GPU detection is active
- [ ] Spot instance interruption handling is configured (for training)
- [ ] Grafana alerts are set for thermal throttling, Xid errors, and vRAM > 95%

---

*Next: [09. GPU Security, Compliance, and Multi-Tenancy](./09_GPU_Security_Compliance.md)*
