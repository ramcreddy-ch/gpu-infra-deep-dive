# 10. GPU Monitoring Deep Dive & Prometheus Metrics

If you can't observe it, you can't fix it. I've seen teams spend weeks debugging a "slow model" when the root cause was a degrading PCIe link that a single Prometheus metric would have caught. This chapter is my complete monitoring playbook.

**Author:** [Ramchandra Chintala](https://github.com/ramcreddy-ch)

---

## The Monitoring Stack

Here's the stack I deploy on every GPU cluster:

```
GPU Hardware
    │
    ▼
DCGM Exporter (DaemonSet) ──► Prometheus ──► Grafana Dashboards
    │                              │
    ▼                              ▼
NVIDIA NVSMI                  AlertManager ──► PagerDuty / Slack
    │
    ▼
Custom Python Exporter (for app-level GPU metrics)
```

### Why DCGM, Not Just nvidia-smi?

`nvidia-smi` is a CLI tool. It's great for interactive debugging, but it doesn't expose metrics in a Prometheus-compatible format, and it misses deeper hardware counters like ECC errors, NVLink bandwidth, and PCIe replay counts.

DCGM (Data Center GPU Manager) is NVIDIA's daemon-level monitoring layer. It runs as a DaemonSet via the GPU Operator and exposes 100+ metrics.

---

## The Metrics That Actually Matter

I've filtered DCGM's 100+ metrics down to the ones I actually alert on. Most of the others are noise.

### Tier 1: Alert Immediately (Page the on-call)

| Metric | Meaning | Threshold | Why It Matters |
|--------|---------|-----------|---------------|
| `DCGM_FI_DEV_XID_ERRORS` | Hardware error codes | Any value > 0 | Xid 74/79 = GPU is dying. Drain the node. |
| `DCGM_FI_DEV_GPU_TEMP` | Core temp (°C) | > 85°C for 5 min | Thermal throttling destroys throughput silently |
| `DCGM_FI_DEV_POWER_VIOLATION` | Power limit exceeded | > 0 | GPU is being power-throttled |
| `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | PCIe retransmissions | > 100 in 5 min | Physical link is degrading |
| `DCGM_FI_DEV_RETIRED_PAGES_SBE` | Single-bit ECC errors (retired pages) | > 10 | Memory is failing. Node needs replacement |

### Tier 2: Alert Within 1 Hour (Investigate)

| Metric | Meaning | Threshold |
|--------|---------|-----------|
| `DCGM_FI_DEV_FB_USED` | vRAM usage (MB) | > 95% of total |
| `DCGM_FI_DEV_GPU_UTIL` | SM utilization | < 20% for 30 min (wasting money) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | Memory bandwidth util | > 90% (memory-bound workload) |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` | NVLink throughput | < 50% of theoretical max |

### Tier 3: Daily Review (Cost optimization)

| Metric | Meaning | What I Do |
|--------|---------|-----------|
| `DCGM_FI_DEV_GPU_UTIL` | Avg utilization per pod | If < 30% consistently, the pod is over-provisioned |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | Actual compute engine activity | Cross-reference with GPU_UTIL to catch the "fake 100%" lie |

---

## Prometheus Alert Rules

Here are the actual PrometheusRule manifests I deploy:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-hardware-alerts
  namespace: monitoring
spec:
  groups:
    - name: gpu-critical
      interval: 30s
      rules:
        - alert: GPUXidError
          expr: DCGM_FI_DEV_XID_ERRORS > 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "GPU Xid error on {{ $labels.gpu }} (node {{ $labels.Hostname }})"
            description: "Xid {{ $value }} detected. Check dmesg for NVRM errors. May require node drain."
            runbook: "https://wiki.internal/gpu-runbooks/xid-errors"

        - alert: GPUThermalThrottling
          expr: DCGM_FI_DEV_GPU_TEMP > 83
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "GPU {{ $labels.gpu }} at {{ $value }}°C — throttling active"

        - alert: GPUMemoryFull
          expr: (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) * 100 > 95
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "GPU {{ $labels.gpu }} vRAM at {{ $value }}% — OOM imminent"

        - alert: GPUPCIeErrors
          expr: rate(DCGM_FI_DEV_PCIE_REPLAY_COUNTER[5m]) > 20
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "PCIe replay errors on GPU {{ $labels.gpu }}. Physical link degrading."

    - name: gpu-cost
      interval: 5m
      rules:
        - alert: GPUIdleWaste
          expr: DCGM_FI_DEV_GPU_UTIL < 5 and on(pod) kube_pod_status_phase{phase="Running"} == 1
          for: 2h
          labels:
            severity: warning
          annotations:
            summary: "GPU {{ $labels.gpu }} in pod {{ $labels.pod }} idle for 2+ hours"

        - alert: GPUECCErrorsAccumulating
          expr: DCGM_FI_DEV_RETIRED_PAGES_SBE > 10
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "GPU {{ $labels.gpu }} has {{ $value }} retired memory pages. Hardware replacement needed."
```

---

## Grafana Dashboard Panels

Here's the JSON structure for the GPU dashboard I use. I broke it into rows that match how I actually troubleshoot.

### Row 1: Fleet Overview
- **Panel:** GPU Utilization Heatmap (all GPUs across all nodes)
- **Panel:** vRAM Usage per GPU (stacked bar chart)
- **Panel:** Total GPUs Available vs. Allocated (single stat)

### Row 2: Per-GPU Deep Dive
- **Panel:** SM Clock Speed over time (catches throttling)
- **Panel:** Temperature over time
- **Panel:** Power draw over time
- **Panel:** PCIe Bandwidth utilization

### Row 3: Training Job Metrics
- **Panel:** GPU utilization per training job (by pod label)
- **Panel:** NCCL all-reduce time (if instrumented)
- **Panel:** Training loss curve (from MLflow)

### Row 4: Serving/Inference Metrics
- **Panel:** Inference latency P50/P95/P99
- **Panel:** Requests per second
- **Panel:** KV Cache utilization (for LLM serving)

---

## Custom Application-Level GPU Metrics

DCGM gives you hardware metrics. But you also need application-level metrics from inside your training/serving code.

Here's a snippet I add to every training script:

```python
"""Custom Prometheus metrics for GPU training jobs."""
from prometheus_client import Gauge, Histogram, start_http_server
import torch
import time

# Define metrics
gpu_memory_allocated = Gauge(
    'training_gpu_memory_allocated_bytes',
    'GPU memory allocated by PyTorch',
    ['gpu_id']
)
gpu_memory_reserved = Gauge(
    'training_gpu_memory_reserved_bytes',
    'GPU memory reserved by PyTorch (includes fragmentation)',
    ['gpu_id']
)
training_step_duration = Histogram(
    'training_step_duration_seconds',
    'Time per training step',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
training_loss = Gauge('training_loss', 'Current training loss')

def update_gpu_metrics():
    """Call this after every training step."""
    for i in range(torch.cuda.device_count()):
        gpu_memory_allocated.labels(gpu_id=str(i)).set(
            torch.cuda.memory_allocated(i)
        )
        gpu_memory_reserved.labels(gpu_id=str(i)).set(
            torch.cuda.memory_reserved(i)
        )

# Start metrics server on port 9090
start_http_server(9090)
```

The difference between `memory_allocated` and `memory_reserved` is crucial:
- **Allocated:** Memory actively used by tensors
- **Reserved:** Memory PyTorch has claimed from CUDA but isn't actively using (fragmentation)

If reserved >> allocated, you have a memory fragmentation problem. The fix is usually calling `torch.cuda.empty_cache()` periodically or reducing batch size.

---

## The Single Command That Saves Hours

When something goes wrong and I need a quick snapshot of everything GPU-related on a node:

```bash
#!/bin/bash
# gpu-snapshot.sh — Run on any GPU node for instant diagnostics
echo "=== GPU Summary ==="
nvidia-smi

echo -e "\n=== Temperature & Throttling ==="
nvidia-smi -q -d CLOCK,TEMPERATURE | grep -E "(GPU Current|GPU Shutdown|Clocks Throttle)"

echo -e "\n=== ECC Errors ==="
nvidia-smi -q -d ECC | grep -E "(Volatile|Aggregate|Retired)"

echo -e "\n=== PCIe Link Status ==="
nvidia-smi -q -d PCIE | grep -E "(Link Gen|Link Width|Replay)"

echo -e "\n=== NVLink Status ==="
nvidia-smi nvlink --status 2>/dev/null || echo "No NVLink on this GPU"

echo -e "\n=== Processes Holding GPU ==="
fuser -v /dev/nvidia* 2>/dev/null

echo -e "\n=== Kernel GPU Errors (last 50 lines) ==="
dmesg -T | grep -i "nvrm\|nvidia\|xid" | tail -50
```

I keep this script on every GPU node and run it as the first step in any incident.

---

*This completes the core documentation. See the [README](../README.md) for the full table of contents.*
