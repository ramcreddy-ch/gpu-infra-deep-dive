# 03. Real-Time Issues & Troubleshooting

In hyper-scale environments, hardware does not fail cleanly. GPUs enter weird states, freeze entire nodes, silently drop PCIe packets, or throw incomprehensible stack traces. Here is the operational runbook used by certified AI engineers.

---

## 🟢 Basic: Identifying the Symptom

When a training or inference job crashes on a GPU, it usually exhibits one of three basic symptoms:
1.  **The Silent OOM:** PyTorch throws a `CUDA Out of Memory` error instantly upon startup.
2.  **The Frozen Node:** The job logs epochs smoothly, and suddenly the logs stop. The GPU utilization drops to 0%, but the process appears "running."
3.  **The Hard Crash:** The container crashes, citing segmentation faults, illegal memory access, or unresolvable library mismatches.

```mermaid
graph TD
    A[GPU Job Fails] --> B{Symptom Type}
    B -->|Instant OOM| C[Zombie Processes]
    B -->|Logs Stop / 0% Util| D[Network Topology / NCCL Timeout]
    B -->|Segfault / Hard Crash| E[Hardware / Dmesg Xid Error]
    
    C --> F((Run $ fuser -v /dev/nvidia*))
    D --> G((Enable NCCL_DEBUG=INFO))
    E --> H((Run dmesg -T | grep NVRM))
```

---

## 🟡 Intermediate: Kernel Panics and Dmesg Decoding

If the issue is not bad PyTorch code, the first place a Platform Engineer looks is the Linux kernel logs. The NVIDIA Driver (`NVRM`) writes hardware-level panics to the kernel ring buffer as **Xid errors**.

Run: `dmesg -T | grep NVRM`

### The Big Three Xid Errors:
*   **Xid 13 (Graphics Engine Exception):**
    *   *Meaning:* The PyTorch workload segfaulted the GPU kernel (like trying to access array index 100 in an array of size 10).
    *   *Troubleshoot:* Rerun PyTorch with `CUDA_LAUNCH_BLOCKING=1`. This forces the GPU to execute synchronously, revealing the exact Python line of code that caused the crash.
*   **Xid 31 (Memory Page Fault):**
    *   *Meaning:* The GPU tried to access a memory space that hasn't been mapped. Often caused when the CPU "un-pins" memory while the GPU DMA engine is actively reading it.
    *   *Troubleshoot:* Check container teardown grace periods and Dataloader `pin_memory=True` hygiene.
*   **Xid 74 / 79 (Fallen off the bus):**
    *   *Meaning:* Lethal hardware failure. The PCIe riser cable is warping from 85°C heat, or the NVLink bridge is damaged.
    *   *Troubleshoot:* Your K8s cluster automation should instantly taint the node (`kubectl taint nodes node-X nvidia.com/gpu=dead:NoSchedule`), drain it, and trigger a vendor RMA.

---

## 🔴 Advanced: Zombie Processes and NCCL Fabrics

### 1. Reaping GPU Zombies
If `nvidia-smi` shows 80GB of memory used, but 0 running processes in the bottom pane, the memory is marooned. 
This happens when K8s force-kills an out-of-control container (`SIGKILL`), but the driver refused to drop the memory pointer to prevent a kernel panic. 

**Expert Fix:**
You must manually find the hidden PIDs locking the device driver.
```bash
# 1. Reveal hidden PIDs locking the hardware
fuser -v /dev/nvidia*

# 2. Aggressive cleanse
kill -9 $(fuser -v /dev/nvidia* 2>/dev/null)

# 3. Extreme reset (Drops all active jobs, purges state)
sudo nvidia-smi -r
```

### 2. Multi-Node Networking: NCCL Timeouts
Training across multiple servers relies on the **NVIDIA Collective Communications Library (NCCL)**. If one GPU falls slightly behind (thermal throttling) or a network link drops a packet, all other GPUs will sit at a barrier wait endlessly. Your cluster freezes.

**Enabling NCCL Traces:**
Inject these flags into the container environment:
`NCCL_DEBUG=INFO`
`NCCL_DEBUG_SUBSYS=INIT,NET`

Look closely at the resulting pod logs:
*   **The Good Log:** `NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE` (It found your 400Gbps InfiniBand card).
*   **The Catastrophic Log:** `NCCL INFO NET/Socket : Using [0]eth0` (It failed to find InfiniBand, and fell back to standard TCP ethernet. Your job will take 10x longer).

*Fix:* Sometimes Kubernetes CNI networks (Flannel, Calico) confuse NCCL, causing it to use a virtual IP gateway. Force the binding explicitly:
`NCCL_IB_HCA=mlx5_0,mlx5_1` (Binds only to the physical Mellanox NICs).
