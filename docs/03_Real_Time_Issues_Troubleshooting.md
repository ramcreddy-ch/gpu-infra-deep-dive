# 03. Real-Time Issues & Troubleshooting

In a real-world MLOps environment, GPUs do not fail gracefully. They enter weird states, freeze entire nodes, or silently corrupt data. Below are the most common production issues I have root-caused and resolved across thousands of A100s/H100s.

## 1. The Undead: GPU Zombie Processes
**Symptoms:** 
You try to run a workload and instantly receive `CUDA Out of Memory`. However, `nvidia-smi` shows no active processes using the GPU, but the memory usage bar is at 80GB/80GB.

**The Root Cause:**
A PyTorch or TensorFlow container was killed via `SIGKILL` (OOMKilled by Kubernetes) or crashed ungracefully. The GPU driver was not notified to release the memory allocation. The process is dead in Linux but alive in the NVIDIA drivers.

**The Fix:**
1. Do NOT reboot the node immediately.
2. Find the lingering process IDs stuck in the GPU memory using `fuser -v /dev/nvidia*`.
3. Force kill them: `kill -9 $(fuser -v /dev/nvidia* 2>/dev/null)`.
4. If that fails, execute a GPU reset (requires dropping active workloads): `sudo nvidia-smi -r`.

## 2. ECC Errors & Uncorrectable PCIe AER
**Symptoms:** 
Training jobs randomly crash with `NCCL WARN: Cuda failure` or `Transport mismatch`. Deep into `dmesg`, you see `NVRM: Xid (PCI:0000:01:00): 62, Internal Timer Reset`.

**The Root Cause:**
Hardware degradation or bus instability. GPUs operate at extreme thermals. Over time, Memory modules (HBM) develop stuck bits (ECC errors) or the PCIe riser cables slightly warp, causing Advanced Error Reporting (AER) resets.

**The Fix:**
1. Check ECC states: `nvidia-smi -q -d ECC`. If you see "Uncorrectable Errors", the GPU must be RMA'd or cordoned from the K8s cluster.
2. To cordon a bad node in K8s: `kubectl cordon node-xyz` and notify hardware ops.
3. Drain the node gracefully to protect running workloads on healthy GPUs on that same node.

## 3. NCCL Timeouts and Multi-Node Freezes
**Symptoms:** 
Job logs show epochs progressing normally, but suddenly the job stops printing logs. GPU utilization drops to 0%, but the processes are still "running". 

**The Root Cause:**
NCCL (NVIDIA Collective Communications Library) timeout. Multi-node distributed training relies on all GPUs syncing up (barrier). If one GPU falls behind (thermal throttling, slow network link, disk I/O lag), all other GPUs will wait endlessly at the barrier for it, freezing the training run.

**The Fix:**
1. Set the NCCL Debug flag to find the rogue node: `export NCCL_DEBUG=INFO`
2. Test network fabric manually. Look for flapping InfiniBand links using `ibstat`.
3. Check for thermal throttling on the delayed node: `nvidia-smi -q -d THERMAL`.

## 4. Thermal Throttling
**Symptoms:** 
Performance incrementally degrades over 3 hours. TeraFLOPS output drops by 30-50%.

**The Root Cause:**
The server fans cannot dispel the heat fast enough. The GPU hits its "Slowdown Temp" (usually 85°C) and automatically underclocks its cores to prevent physical melting.

**The Fix:**
1. Check clock caps: `nvidia-smi -q -d CLOCK`. Look for `HW Slowdown: Active`.
2. Inspect data center cooling or container density. Do not pack 8 intensive workloads on adjacent physical slots if the chassis cooling isn't rated for it.
