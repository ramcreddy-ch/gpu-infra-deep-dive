# NCCL, InfiniBand, RDMA, NVLink, NVSwitch & High-Speed Networking

> *In distributed AI, compute is cheap; moving data is expensive. If your network is not configured perfectly, your $40,000 H100 GPUs will sit idle 90% of the time waiting for data. Network engineering for AI is entirely different from traditional web networking.*

---

## 1. What Problem Does This Solve?

### The Communication Bottleneck

During distributed training (DDP or FSDP), every GPU must synchronize its gradients with every other GPU.
If you have 1,024 GPUs training a 70B model, you must move 140 GB of gradients across the network at the end of *every single step*.

**Traditional Networking (TCP/IP over Ethernet):**
1. GPU sends data to CPU RAM over PCIe.
2. CPU copies data from RAM to kernel space.
3. OS TCP stack breaks data into packets, adds headers, computes checksums.
4. CPU copies packets to NIC (Network Interface Card) over PCIe.
5. NIC sends packets over the wire.
6. Receiving side does the exact reverse.

**Why this fails for AI:**
- **High Latency:** TCP overhead adds ~20-50 microseconds. In AI, we need < 2 μs.
- **CPU Bottleneck:** Pushing 400 Gbps of TCP traffic will consume 100% of all CPU cores, starving the dataloaders.
- **Bandwidth Limits:** Going through system RAM halves your effective bandwidth.

### The Solution: RDMA and GPU-Direct

RDMA (Remote Direct Memory Access) allows the NIC on Server A to read/write directly to the RAM on Server B, completely bypassing the CPU and OS kernel.
**GPU-Direct RDMA** takes this a step further: The NIC on Server A reads directly from the **GPU VRAM** on Server A and writes directly to the **GPU VRAM** on Server B.

---

## 2. Internal Architecture

### The Hardware Topology (Single Node)

```
┌─────────────────────────────────────────────────────────────┐
│                 HGX H100 Node (8 GPUs)                      │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                NVSwitch Fabric (3.6 TB/s)             │  │
│  └─┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬────┘  │
│    │NVL   │NVL   │NVL   │NVL   │NVL   │NVL   │NVL   │NVL    │
│  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐     │
│  │GPU│  │GPU│  │GPU│  │GPU│  │GPU│  │GPU│  │GPU│  │GPU│     │
│  │ 0 │  │ 1 │  │ 2 │  │ 3 │  │ 4 │  │ 5 │  │ 6 │  │ 7 │     │
│  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘     │
│    │PCIe  │PCIe  │PCIe  │PCIe  │PCIe  │PCIe  │PCIe  │PCIe   │
│  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐     │
│  │NIC│  │NIC│  │NIC│  │NIC│  │NIC│  │NIC│  │NIC│  │NIC│     │
│  │ 0 │  │ 1 │  │ 2 │  │ 3 │  │ 4 │  │ 5 │  │ 6 │  │ 7 │     │
│  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘     │
│    │      │      │      │      │      │      │      │       │
└────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼───────┘
     │      │      │      │      │      │      │      │
     ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
    To InfiniBand / RoCE ToR (Top of Rack) Switches
    (400 Gbps per NIC = 3.2 Tbps aggregate out-of-node bandwidth)
```

**Key Concept: The Rail-Optimized Network**
Notice there are 8 GPUs and 8 NICs (Network Interface Cards, typically ConnectX-7).
NIC 0 is connected to the same PCIe switch as GPU 0.
In a "rail-optimized" topology, NIC 0 on Node A only talks to NIC 0 on Node B. This creates a straight, unimpeded "rail" for data to travel without crossing the CPU or QPI/UPI links.

### The Software Layer: NCCL

NVIDIA Collective Communications Library (NCCL - pronounced "Nickel") is the software that orchestrates this hardware. PyTorch does not know how to route packets; it simply says `dist.all_reduce(tensor)`.

**What NCCL does internally:**
1. **Topology Discovery:** When initialized, NCCL reads the PCIe tree (sysfs) to map which GPUs are attached to which NICs and CPU sockets.
2. **Algorithm Selection:** Based on the topology, it selects an algorithm.
   - For intra-node (GPU 0 to GPU 1): It uses **NVLink**.
   - For inter-node (Node A to Node B): It uses **GPU-Direct RDMA** via the NIC.
3. **Ring / Tree Construction:** It forms a logical ring or tree across all 1,024 GPUs to pass the data efficiently without bottlenecks.

---

## 3. Deep Internal Working: InfiniBand vs RoCE

To achieve RDMA between nodes, you have two protocol choices:

### InfiniBand (IB)
- **The Standard for AI:** Used by OpenAI, Meta (RSC), NVIDIA (SuperPOD).
- **Architecture:** A completely separate networking stack from Ethernet. Requires IB NICs, IB switches (Quantum-2), and IB cables.
- **Why it's better:** It is natively **lossless**. It uses a credit-based flow control system. A switch will not transmit a packet unless it knows the receiving switch has buffer space. Result: Zero dropped packets, ultra-low latency (~1 μs).

### RoCE v2 (RDMA over Converged Ethernet)
- **The Challenger:** Used by hyperscalers who refuse to build a separate IB network (AWS, Azure, Meta for some clusters).
- **Architecture:** Encapsulates IB packets inside standard UDP/IP Ethernet packets.
- **The Problem:** Ethernet is natively **lossy** (it drops packets when congested).
- **The Fix (PFC & ECN):** To make RoCE work, you must configure Priority Flow Control (PFC) and Explicit Congestion Notification (ECN) on your Arista/Cisco switches. This is notoriously difficult to tune. A single misconfigured switch port will cause PFC storms, bringing the entire AI cluster to a halt.

---

## 4. Production Architecture

### The Spine-Leaf Fat-Tree Topology

For massive training clusters (e.g., 4,000 GPUs), the network must be **non-blocking**. If all 4,000 GPUs transmit at 400 Gbps simultaneously, the network must not drop a single packet.

```
                  ┌─────────┐   ┌─────────┐   ┌─────────┐
    Spine Tier    │ Spine 1 │   │ Spine 2 │...│Spine N  │ (Director Switches)
                  └─┬─┬─┬─┬─┘   └─┬─┬─┬─┬─┘   └─┬─┬─┬─┬─┘
                    │ │ │ │       │ │ │ │       │ │ │ │
        ┌───────────┘ │ │ └───────┐ │ │ └───────┐ │ │ └────────┐
        │             │ │         │ │ │         │ │ │          │
      ┌─▼───────┐   ┌─▼───────┐   ┌─▼───────┐   ┌─▼───────┐
Leaf  │ Leaf 1  │   │ Leaf 2  │   │ Leaf 3  │   │ Leaf 4  │ (ToR Switches)
Tier  └─┬─┬─┬─┬─┘   └─┬─┬─┬─┬─┘   └─┬─┬─┬─┬─┘   └─┬─┬─┬─┬─┘
        │ │ │ │       │ │ │ │       │ │ │ │       │ │ │ │
        ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼       ▼ ▼ ▼ ▼
      [ Node 1 ]    [ Node 2 ]    [ Node 3 ]    [ Node 4 ]
      (8x H100)     (8x H100)     (8x H100)     (8x H100)
```

**Bisection Bandwidth:** In a full fat-tree, the bisection bandwidth is 1:1. You can cut the cluster in half, and the two halves can communicate at full line rate.

---

## 5. Production Incident Scenarios

### Incident 1: "NCCL Falls Back to CPU / PCIe"
**Symptoms:** Multi-node training is 10x slower than expected. CPU usage is pinned at 100%.
**Root Cause:** NCCL could not establish an RDMA connection between the nodes. When RDMA fails, NCCL silently falls back to sending data over standard TCP/IP sockets via the CPU.
**Fix:** 
1. Check NCCL logs: `NCCL_DEBUG=INFO`. Look for `Using [X] via IP/TCP` instead of `via NET/IB/0`.
2. Ensure the InfiniBand/RoCE interface is correct: `export NCCL_SOCKET_IFNAME=eth1`.
3. Check MOFED drivers and ensure `ibv_devinfo` shows active ports.

### Incident 2: "PFC Pause Storm (RoCE v2)"
**Symptoms:** The entire Kubernetes cluster networking freezes. Liveness probes fail, pods restart.
**Root Cause:** A microburst of AI traffic caused a switch buffer to fill up. The switch sent a PFC Pause frame to the sender. The sender paused, causing its own buffers to fill up, so it sent a Pause frame to *its* sender. This cascaded across the spine-leaf architecture, locking up the entire fabric (Head-of-Line blocking).
**Fix:** Tune switch ECN (Explicit Congestion Notification) thresholds to trigger *before* PFC is necessary. ECN tells the sender to slow down gracefully (like TCP windowing) rather than slamming on the brakes.

### Incident 3: "NUMA Misalignment Bottleneck"
**Symptoms:** Peak bandwidth between nodes is capped at 150 GB/s instead of 400 GB/s.
**Root Cause:** The training pod was scheduled on CPU Socket 0, but it was using the NIC attached to CPU Socket 1. The RDMA traffic had to cross the QPI/UPI link between the CPUs, which became the bottleneck.
**Fix:** Use Kubernetes Topology Manager to ensure Pods, GPUs, and SR-IOV NIC virtual functions are allocated on the exact same NUMA node.

---

## 6. Performance Optimization

### Environment Variable Tuning for NCCL

```bash
# Critical NCCL tuning for production clusters
export NCCL_DEBUG=INFO                   # Turn this on when debugging, WARN for prod
export NCCL_IB_DISABLE=0                 # Ensure IB/RoCE is enabled
export NCCL_NET_GDR_LEVEL=5              # Force GPU-Direct RDMA
export NCCL_IB_GID_INDEX=3               # Required for RoCE v2 routing
export NCCL_IB_TC=106                    # Traffic Class mapping for QoS/PFC
export NCCL_ALGO=Ring                    # Force Ring topology (sometimes better than Tree)
```

### NCCL Tests (The Benchmark)

Before running PyTorch, you must validate the fabric using `nccl-tests`. If `nccl-tests` is slow, PyTorch will be slow.

```bash
# Run an AllReduce benchmark across 2 nodes (16 GPUs)
mpirun -np 16 -H node1:8,node2:8 \
  -x NCCL_DEBUG=INFO \
  /opt/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1

# Look for the "Out of place algbw (GB/s)" column.
# On 400Gbps NDR, you should see ~45 GB/s algorithm bandwidth per GPU.
```

---

## Summary

```
Networking Cheat Sheet:

| Technology | Speed | Domain | Purpose |
|---|---|---|---|
| **PCIe Gen5** | 128 GB/s | Within Node | CPU ↔ GPU or NIC ↔ GPU |
| **NVLink 4.0** | 900 GB/s | Within Node | GPU ↔ GPU (Direct) |
| **NVSwitch** | 3.6 TB/s | Within Node | Connects all 8 GPUs in a mesh |
| **InfiniBand** | 400 Gbps | Across Nodes | Lossless, ultra-low latency GPU ↔ GPU |
| **RoCE v2** | 400 Gbps | Across Nodes | IB over Ethernet, requires QoS tuning |

If NCCL hangs → It's a network timeout (check IB links, switch logs).
If training is slow → Check if NCCL fell back to TCP (NCCL_DEBUG=INFO).
```

---
*Next: [09 — Multi-GPU and Multi-Node Inference →](09_Multi_GPU_Multi_Node_Inference.md)*
