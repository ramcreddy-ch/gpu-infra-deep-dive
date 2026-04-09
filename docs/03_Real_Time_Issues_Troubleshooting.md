# 03. Real-Time Issues & Troubleshooting (10/10 Enterprise Depth)

In a hyper-scale cluster, GPUs do not fail gracefully. They enter weird states, freeze entire nodes, silently corrupt tensors, or throw incomprehensible stack traces. Here is the operational runbook used by certified engineers.

---

## 1. Hardware Errors: Decoding Xid Events

When a Deep Learning job crashes randomly (sometimes citing "illegal memory access" or "CUDA launch failure"), the first place a Platform Engineer looks is not testing Python code, it is the Linux kernel logs.

Run: `dmesg -T | grep NVRM`
If you have a hardware issue, you will see an `Xid` error in the system logs.

**Common Xid Panics and Resolutions:**

*   **Xid 13: Graphics Engine Exception.**
    *   *Cause:* The PyTorch workload segfaulted the GPU kernel. Usually an out-of-bounds array access in a custom CUDA kernel, or severe memory corruption.
    *   *Fix:* Run the workload with `CUDA_LAUNCH_BLOCKING=1` to force synchronous execution. This forces PyTorch to print the exact Python line that caused the GPU trap.
*   **Xid 31: Memory Page Fault.**
    *   *Cause:* The GPU tried to access a memory space that hasn't been mapped. Often happens if host pinned memory (CPU RAM) is aggressively unmapped while the GPU DMA engine is still copying data.
    *   *Fix:* Check PyTorch DataLoader `pin_memory=True` usage and ensure process shutdown signals (SIGTERM) are handled cleanly.
*   **Xid 74 or Xid 79: Fallen off the bus / NVLink Error.**
    *   *Cause:* Lethal hardware failure. The PCIe riser is warping from 85°C heat, or an NVLink bridge has physical micro-fractures.
    *   *Log Snippet:* `NVRM: GPU at PCI:0000:41:00: GPU-bfxxx-xxx fallen off the bus.`
    *   *Fix:* Cluster automation should instantly Taint the Kubernetes node (`kubectl taint nodes node-xyz nvidia.com/gpu=dead:NoSchedule`), drain it, and trigger a vendor RMA.

---

## 2. Multi-Node Networking: The NCCL Debug Runbook

Training across multiple nodes relies exclusively on the **NVIDIA Collective Communications Library (NCCL)**. If NCCL fails, the job hangs endlessly because all GPUs wait at a barrier synchronization perfectly at 0% GPU utilization.

### Forcing NCCL Traces
Inject these environment variables into your Kubernetes Pod / Job manifest:
```yaml
env:
  - name: NCCL_DEBUG
    value: "INFO" # or "TRACE" for extreme depth
  - name: NCCL_DEBUG_SUBSYS
    value: "INIT,NET"
```

Look at the Pod logs:
*   **Good Log:** `NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE`
*   **Bad Log (Fallback):** `NCCL INFO NET/Socket : Using [0]eth0`
    *   *Why this is bad:* The framework failed to find the InfiniBand/RDMA hardware. It has fallen back to standard ethernet TCP sockets. Training time will inflate from 2 days to 30 days.

### Enforcing Network Interfaces
Sometimes virtual networking (e.g., Flannel, Calico) creates virtual bridge interfaces (e.g. `docker0`, `cni0`) that confuse NCCL. NCCL attempts to route traffic over a 10Gbps virtual IP instead of your 400Gbps RDMA card.
*   *Fix:* Restrict NCCL's interface search explicitly:
    `NCCL_SOCKET_IFNAME=eth0` or `NCCL_IB_HCA=mlx5_0,mlx5_1`

### InfiniBand Port Flapping
If the network itself is dying at the physical layer, check the Mellanox ConnectX hardware on the host:
```bash
$ ibstat
CA 'mlx5_0'
        port 1:
                State: Active
                Physical state: LinkUp
                Rate: 100 Gb/sec (4X EDR)
```
If `State` says `Down` or `Physical state` says `Polling`, your optical transceiver cable is unplugged, dirty, or burning out.

---

## 3. The Undead: Zombie GPU Processes

**Symptoms:** `nvidia-smi` shows 80GB memory used, but 0 running processes in the bottom pane. "CUDA OOM".

**Root Cause:** A container was OOMKilled by Kubernetes (`SIGKILL`), but the driver was heavily utilizing the GPU DMA engine and refused to drop the memory pointer allocation to prevent a kernel panic. The memory is marooned.

**The Fix:**
You must manually reap the zombies by searching the device file locks.
(We have included `scripts/zombie_killer.sh` in this repository specifically to automate this).
```bash
# Reveal the hidden PIDs locking the GPU device drivers
fuser -v /dev/nvidia*

# Aggressive cleanse
kill -9 $(fuser -v /dev/nvidia* 2>/dev/null)

# Extreme reset (purges all state, aborts all workloads on the card)
sudo nvidia-smi -r
```
