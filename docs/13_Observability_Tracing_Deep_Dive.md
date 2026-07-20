# Observability: OpenTelemetry, Prometheus, Grafana, Langfuse & Arize Phoenix

> *Traditional APM tools track CPU, RAM, and HTTP latency. In AI infrastructure, you must track GPU temperature, Tensor Core utilization, KV cache swap rates, token generation latency, and semantic drift. If you cannot observe the silicon and the model simultaneously, you cannot run production AI.*

---

## 1. What Problem Does This Solve?

### The Blind Spot of AI

When an LLM endpoint slows down, standard tools (Datadog/NewRelic) tell you: "HTTP POST /generate took 4.5 seconds."

That is useless for an AI Engineer. To fix the issue, you need to know:
1. **Hardware:** Was the GPU thermal throttling?
2. **System:** Was vLLM swapping KV cache to the CPU?
3. **Model:** Was the prompt 8,000 tokens long?
4. **Agent:** Did the LLM decide to call an external tool (Search) that timed out?

To solve this, AI Observability requires a **full-stack approach** from the physical GPU die up to the semantic meaning of the text.

---

## 2. Internal Architecture

### Full-Stack AI Observability Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Observability Stack                              │
│                                                                         │
│  ┌──────────────────────┐  ┌────────────────────┐  ┌─────────────────┐  │
│  │ Application Tracing  │  │ LLM Engine Metrics │  │ GPU Telemetry   │  │
│  │ (Langfuse / Phoenix) │  │ (Prometheus)       │  │ (DCGM Exporter) │  │
│  └──────────┬───────────┘  └─────────┬──────────┘  └────────┬────────┘  │
│             │                        │                      │           │
│             ▼                        ▼                      ▼           │
│  ┌──────────────────────┐  ┌─────────────────────────────────────────┐  │
│  │ Traces & Evaluation  │  │ Time-Series Database (Prometheus)       │  │
│  │ (OpenTelemetry / DB) │  │ • vllm:num_requests_waiting             │  │
│  └──────────┬───────────┘  │ • dcgm:gpu_temp                         │  │
│             │              └─────────┬───────────────────────────────┘  │
│             ▼                        ▼                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                           Grafana                                 │  │
│  │  [ GPU Heatmap ]  [ Token Latency ]  [ KV Cache Usage ]           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Internal Working

### 1. Hardware Observability (DCGM)

NVIDIA Data Center GPU Manager (DCGM) is a suite of tools for cluster management. The `dcgm-exporter` exposes hardware metrics to Prometheus.

**Critical DCGM Metrics for AI:**
- `DCGM_FI_DEV_GPU_UTIL`: % of time a kernel was executing.
- `DCGM_FI_PROF_SM_ACTIVE`: % of time Streaming Multiprocessors had at least one warp active.
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`: % of time Tensor Cores were doing matrix math.
- `DCGM_FI_DEV_GPU_TEMP`: GPU temperature.
- `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`: Bitmask explaining *why* the GPU is slow (Thermal, Power, Sync).

### 2. System Observability (vLLM Metrics)

vLLM exposes Prometheus metrics at `/metrics`.

**Critical vLLM Metrics for AI:**
- `vllm:num_requests_running`: Batch size currently executing on GPU.
- `vllm:num_requests_waiting`: Queue depth (requires autoscaling if > 0).
- `vllm:gpu_cache_usage_perc`: KV cache saturation.
- `vllm:time_to_first_token_seconds`: Prefill latency (TTFT).
- `vllm:time_per_output_token_seconds`: Decode latency (TPOT).

### 3. Application Observability (LLM Tracing)

When using complex chains (e.g., LangChain, LlamaIndex), a single user request might trigger 5 different LLM calls and 3 Vector DB searches. OpenTelemetry (OTel) spans are used to trace the execution graph.

Tools like **Langfuse** or **Arize Phoenix** ingest these OTel traces to visualize:
- The exact prompt sent at step 3.
- The cost ($) of the tokens consumed.
- The retrieval latency of the Vector DB.

---

## 4. Production Architecture

### Instrumenting an AI Gateway

```python
# Instrumenting an LLM application with OpenTelemetry and Arize Phoenix
import phoenix as px
from opentelemetry import trace
from openinference.instrumentation.openai import OpenAIInstrumentor

# 1. Start Phoenix backend (local or cloud)
px.launch_app()

# 2. Instrument OpenAI API calls (auto-captures prompts, tokens, latency)
OpenAIInstrumentor().instrument()

# 3. Your normal application code
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing."}]
)
# Phoenix automatically records a Trace Span containing the prompt, 
# token count, model name, and generation time.
```

---

## 5. Production Incident Scenarios

### Incident 1: "100% GPU Utilization but Slow Generation"
**Symptoms:** `nvidia-smi` shows 100% GPU-Util. But tokens are generating at 5 tok/sec on an A100 (should be ~40).
**Root Cause:** The `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` metric in Grafana showed 0%. The `DCGM_FI_PROF_PIPE_FP16_ACTIVE` showed 0%. The model was quantized to a format (e.g., 3-bit) that required the GPU to use standard CUDA Cores for integer math rather than Tensor Cores. The GPU was working incredibly hard, but doing the math inefficiently.
**Fix:** Switch to an FP8 or INT4 AWQ model that can leverage Tensor Cores.

### Incident 2: "The Silent Queue Explosion"
**Symptoms:** Users complain about 30-second delays before the model starts typing. `vllm:time_per_output_token` (generation speed) is fast (20ms), but `vllm:time_to_first_token` is 30 seconds.
**Root Cause:** The Grafana panel for `vllm:num_requests_waiting` showed a spike to 150. The KV cache was 99% full, so vLLM stopped admitting new requests and queued them. The autoscaler (KEDA) was watching CPU usage (which was low) instead of watching the vLLM queue depth.
**Fix:** Change the KEDA trigger to scale the vLLM deployment based on the Prometheus metric `vllm:num_requests_waiting > 10`.

### Incident 3: "RAG Prompt Drift"
**Symptoms:** Over a month, the cost of the OpenAI API tripled, even though user traffic remained flat.
**Root Cause:** Looking at Langfuse traces, the average input prompt length grew from 2,000 tokens to 8,000 tokens. An engineer updated the Vector DB retrieval logic to return `top_k=20` instead of `top_k=5` without telling anyone.
**Fix:** Set up an anomaly detection alert on the `llm.usage.prompt_tokens` metric.

---

## 6. Performance Optimization

### SLIs and SLOs for AI

Site Reliability Engineering (SRE) for AI requires new Service Level Indicators (SLIs).

| Traditional Web SLI | AI Infrastructure SLI |
|---|---|
| Request Latency (ms) | Time To First Token (TTFT) |
| Throughput (Req/sec) | Time Per Output Token (TPOT) |
| Error Rate (HTTP 500) | Generation Failure Rate (OOMs, timeouts) |
| CPU Utilization | Tensor Core Utilization / KV Cache Saturation |

**Example AI SLO:**
"95% of requests must achieve a TTFT < 500ms and a TPOT < 50ms, evaluated over a 7-day window."

---

## 7. Kubernetes / Implementation Perspective

### Deploying the Prometheus Stack for GPUs

```yaml
# Prometheus ServiceMonitor to scrape vLLM
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-monitor
spec:
  endpoints:
  - port: http
    path: /metrics
    interval: 10s
  selector:
    matchLabels:
      app: vllm-inference
```

---

## Summary

```
The AI Observability Golden Signals:
1. Hardware: Tensor Core Activity %, Temperature, Clock Throttling
2. Engine: KV Cache Usage %, Waiting Queue Depth
3. Latency: TTFT (Prefill speed), TPOT (Decode speed)
4. Application: Token Counts (Cost), Semantic Traces (Quality)

Tools:
- DCGM + Prometheus + Grafana (Hardware/System)
- Langfuse / Phoenix / Datadog LLM (Application/Semantic)
```

---
*Next: [14 — LLM Evaluation, Benchmarks & Hallucination Detection →](14_LLM_Evaluation_Quality_Deep_Dive.md)*
