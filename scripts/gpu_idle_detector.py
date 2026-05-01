#!/usr/bin/env python3
"""
GPU Idle Pod Detector for Kubernetes
Author: Ramchandra Chintala

Queries Prometheus for DCGM GPU utilization metrics and identifies
pods that have been idle (< threshold) for longer than the grace period.
Outputs a report and optionally deletes idle pods.

Usage:
  python gpu_idle_detector.py --prometheus-url http://prometheus:9090 --threshold 5 --grace-minutes 120
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def query_prometheus(prom_url, query):
    """Execute a PromQL instant query."""
    resp = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")
    return data["data"]["result"]


def find_idle_gpu_pods(prom_url, threshold_pct, grace_minutes):
    """Find GPU pods with utilization below threshold for longer than grace period."""
    # Average GPU utilization per pod over the grace period window
    query = (
        f'avg_over_time(DCGM_FI_DEV_GPU_UTIL{{pod!=""}}[{grace_minutes}m]) < {threshold_pct}'
    )

    results = query_prometheus(prom_url, query)
    idle_pods = []

    for result in results:
        pod = result["metric"].get("pod", "unknown")
        namespace = result["metric"].get("namespace", "unknown")
        gpu_id = result["metric"].get("gpu", "unknown")
        avg_util = float(result["value"][1])

        idle_pods.append({
            "pod": pod,
            "namespace": namespace,
            "gpu": gpu_id,
            "avg_utilization_pct": round(avg_util, 2),
            "idle_window_minutes": grace_minutes,
        })

    return idle_pods


def estimate_waste(idle_pods, gpu_cost_per_hour=3.50):
    """Estimate dollar waste from idle GPU pods."""
    total_hours = 0
    for pod in idle_pods:
        hours = pod["idle_window_minutes"] / 60
        total_hours += hours

    return {
        "idle_gpu_hours": round(total_hours, 1),
        "estimated_waste_usd": round(total_hours * gpu_cost_per_hour, 2),
        "gpu_cost_per_hour": gpu_cost_per_hour,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU Idle Pod Detector")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                        help="Prometheus server URL")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="GPU utilization threshold (percent). Below = idle.")
    parser.add_argument("--grace-minutes", type=int, default=120,
                        help="How long a pod must be idle before flagging (minutes)")
    parser.add_argument("--gpu-cost", type=float, default=3.50,
                        help="Cost per GPU-hour in USD (for waste estimation)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable")
    args = parser.parse_args()

    print(f"Scanning for GPU pods idle (<{args.threshold}% util) for >{args.grace_minutes} min...")
    print(f"Prometheus: {args.prometheus_url}\n")

    try:
        idle_pods = find_idle_gpu_pods(args.prometheus_url, args.threshold, args.grace_minutes)
    except Exception as e:
        print(f"ERROR: Could not query Prometheus: {e}")
        print("Make sure Prometheus is reachable and DCGM Exporter is running.")
        sys.exit(1)

    if not idle_pods:
        print("No idle GPU pods found. All GPUs are being utilized.")
        return

    waste = estimate_waste(idle_pods, args.gpu_cost)

    if args.json:
        print(json.dumps({"idle_pods": idle_pods, "waste_estimate": waste}, indent=2))
        return

    # Human-readable output
    print(f"Found {len(idle_pods)} idle GPU pod(s):\n")
    print(f"{'Pod':<45} {'Namespace':<20} {'GPU':<8} {'Avg Util':<10}")
    print("-" * 85)
    for pod in idle_pods:
        print(f"{pod['pod']:<45} {pod['namespace']:<20} {pod['gpu']:<8} {pod['avg_utilization_pct']:<10}%")

    print(f"\n--- Waste Estimate ---")
    print(f"Idle GPU-hours: {waste['idle_gpu_hours']}")
    print(f"Estimated waste: ${waste['estimated_waste_usd']} (at ${waste['gpu_cost_per_hour']}/GPU-hr)")
    print(f"\nTo reclaim, run: kubectl delete pod <pod-name> -n <namespace>")


if __name__ == "__main__":
    main()
