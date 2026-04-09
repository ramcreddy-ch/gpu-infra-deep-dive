#!/usr/bin/env python3
"""
GPU Health Metrics Exporter for Prometheus
Author: Ramchandra Chintala
Description: Extracts deep hardware metrics (Memory, Thermal limits, PCIe drops)
that are commonly missed by standard container metrics, avoiding silent failures.
"""

import time
import os
try:
    import pynvml
except ImportError:
    print("Please install pynvml: pip install pynvml")
    exit(1)

def initialize_nvml():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    return device_count

def scan_gpu_health(device_index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    name = pynvml.nvmlDeviceGetName(handle)
    
    # 1. Memory Usage
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_mem = mem_info.total / (1024**3)
    used_mem = mem_info.used / (1024**3)
    free_mem = mem_info.free / (1024**3)

    # 2. Temperature & Throttling
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    # Check if GPU is throttling due to heat
    clocks_throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
    is_throttling = (clocks_throttle_reasons & pynvml.nvmlClocksThrottleReasonHwSlowdown) != 0

    # 3. Utilization Rates
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = utilization.gpu
    mem_util = utilization.memory

    print(f"--- GPU {device_index}: {name} ---")
    print(f"Memory: {used_mem:.2f}GB / {total_mem:.2f}GB (Free: {free_mem:.2f}GB)")
    print(f"Utilization: Compute={gpu_util}%, I/O Bandwidth={mem_util}%")
    print(f"Temperature: {temp}°C | Thermal Throttling Active: {is_throttling}")

    # Generate Alerts
    if used_mem / total_mem > 0.95:
        print("[CRITICAL ALERT] VRAM is at 95%+ capacity. Impending OOM risk.")
    if is_throttling:
        print("[CRITICAL ALERT] GPU is thermally throttling. TFLOPS performance is degrading.")

if __name__ == "__main__":
    print("Starting GPU Health Scan...")
    count = initialize_nvml()
    print(f"Found {count} NVIDIA Physical Devices.\n")
    for i in range(count):
        scan_gpu_health(i)
    
    pynvml.nvmlShutdown()
