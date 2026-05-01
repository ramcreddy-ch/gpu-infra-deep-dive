#!/bin/bash
# ------------------------------------------------------------------
# Ramchandra Chintala
# Title: GPU Node Diagnostic Snapshot
# Description: One-command instant diagnostics for any GPU node.
#              Captures temperature, ECC errors, PCIe status, NVLink
#              health, zombie processes, and kernel GPU errors.
# Usage: ./gpu_snapshot.sh
# ------------------------------------------------------------------

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GPU Node Diagnostic Snapshot                               ║"
echo "║  $(date '+%Y-%m-%d %H:%M:%S %Z')                          ║"
echo "║  Hostname: $(hostname)                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo ""
echo "=== 1. GPU Summary ==="
nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits 2>/dev/null | \
  awk -F', ' '{printf "GPU %s: %s | Temp: %s°C | Power: %sW/%sW | VRAM: %s/%s MB | Compute: %s%% | MemBW: %s%%\n", $1,$2,$4,$5,$6,$7,$8,$9,$10}' || \
  echo "ERROR: nvidia-smi not available"

echo ""
echo "=== 2. Temperature & Throttling ==="
nvidia-smi -q -d CLOCK 2>/dev/null | grep -E "(GPU Current Temp|GPU T.Limit Temp|Clocks Throttle Reasons)" | head -20 || echo "N/A"

echo ""
echo "=== 3. ECC Memory Errors ==="
nvidia-smi -q -d ECC 2>/dev/null | grep -E "(Volatile|Aggregate|Retired)" | head -20 || echo "No ECC data (consumer GPU or ECC disabled)"

echo ""
echo "=== 4. PCIe Link Health ==="
nvidia-smi -q -d PCIE 2>/dev/null | grep -E "(Link Gen|Link Width|Replay|Tx Throughput|Rx Throughput)" | head -20 || echo "N/A"

echo ""
echo "=== 5. NVLink Status ==="
nvidia-smi nvlink --status 2>/dev/null || echo "No NVLink available on this GPU model"

echo ""
echo "=== 6. Processes Holding GPU Devices ==="
echo "Device file holders:"
fuser -v /dev/nvidia* 2>/dev/null || echo "No processes holding GPU devices"

echo ""
echo "=== 7. nvidia-smi Process List ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "No compute processes"

echo ""
echo "=== 8. Kernel GPU Errors (last 30 lines) ==="
dmesg -T 2>/dev/null | grep -iE "nvrm|nvidia|xid|gpu|pcie.*error" | tail -30 || echo "No kernel GPU messages found (or insufficient permissions)"

echo ""
echo "=== 9. CUDA Driver & Runtime Versions ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 2>/dev/null || echo "N/A"
if command -v nvcc &>/dev/null; then
    nvcc --version | grep "release" || true
else
    echo "nvcc not found (CUDA toolkit not installed on host)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Snapshot complete. Review output above for anomalies.      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
