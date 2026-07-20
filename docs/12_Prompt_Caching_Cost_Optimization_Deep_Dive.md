# Prompt Caching, Semantic Caching, Context Compression & AI Cost Optimization

> *Inference compute costs will bankrupt an enterprise long before storage or network costs do. Optimization must happen at the architecture level (caching), the system level (KV caches), and the prompt level (compression). A well-optimized AI platform can reduce token processing costs by 80-95%.*

---

## 1. What Problem Does This Solve?

### The Token Cost Multiplier

When an LLM processes a prompt, it performs the "Prefill" phase: it passes the entire prompt through the transformer to build the KV Cache. This takes massive compute (FLOPs).

Imagine a Customer Support Copilot:
- System Prompt + RAG Context: 5,000 tokens.
- User Message: 50 tokens.
- The model processes 5,050 input tokens for every single turn of the conversation.
- If the user asks 10 questions, you pay to process the 5,000-token system prompt 10 times.

If we can **cache** the processing of those 5,000 tokens, or **compress** them into 500 tokens, we cut our compute costs by an order of magnitude.

---

## 2. Internal Architecture

### The Three Layers of AI Caching

```
┌─────────────────────────────────────────────────────────────┐
│                    Enterprise AI Platform                   │
│                                                             │
│  1. Semantic Cache (Redis / Vector DB)                      │
│     • Exact or semantic match of user's query               │
│     • Hit = 0 GPU cycles used. Response in 10ms.            │
│     • Example: "How do I reset my password?"                │
│                                                             │
│  2. Prompt / Prefix Cache (vLLM / SGLang / Bedrock)         │
│     • Caches the KV state of the System Prompt              │
│     • Hit = Prefill Phase skipped. Decode starts instantly. │
│     • Example: "You are a helpful IT assistant..."          │
│                                                             │
│  3. Context Compression (LLMLingua / Summary)               │
│     • Shrinks the RAG context before it hits the LLM        │
│     • Hit = Reduced token count during Prefill              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Internal Working

### Semantic Caching Internals

Semantic caching intercepts the user's prompt *before* it hits the expensive LLM.

1. User sends: *"Can you tell me how to reset my password?"*
2. System embeds this sentence using a fast, cheap embedding model (e.g., BGE-Small on CPU).
3. System queries a Vector Cache (e.g., Redis VL or GPTCache) using Vector Search.
4. The Cache contains a previous query: *"How do I reset my password?"* with the pre-computed LLM answer.
5. The cosine similarity is `0.95`. Since this exceeds the threshold (`0.90`), the system returns the cached answer.

**Cost impact:** $0.00.
**Latency impact:** 20ms instead of 2000ms.

### Prompt Caching (Prefix Caching) Internals

*(Also covered in the vLLM / SGLang chapters, but focused here on cost math).*

When vLLM processes a prompt, it tokenizes it and chunks it into blocks (e.g., 16 tokens). It hashes the token IDs of each block and stores the resulting KV Cache block in GPU memory.

When a new request arrives, vLLM checks the hash table. If the first N blocks match exactly, vLLM simply points the new request's page table to the existing KV blocks in VRAM.

**Why this is a game-changer for cost:**
- Anthropic and OpenAI now offer API-level Prompt Caching.
- Cached input tokens are priced at a **50-80% discount** compared to standard input tokens, because the provider doesn't have to spend GPU compute on the Prefill phase.

### Context Compression Internals (LLMLingua)

If you must pass 10,000 tokens of RAG context, can you compress it?

**Algorithm (LLMLingua):**
1. Pass the context through a small, cheap open-source model (e.g., Llama-3 8B).
2. Look at the perplexity (surprise factor) of each token.
3. Stop-words ("the", "is", "a") have very low perplexity. The model easily predicts them.
4. Nouns and facts have high perplexity.
5. Strip out all low-perplexity tokens. The text becomes unreadable to humans, but the LLM can still understand it perfectly.
6. Pass the compressed 2,000 token context to the expensive model (GPT-4 / Claude Opus).

---

## 4. Production Architecture

### The Cost-Optimized Inference Gateway

```
┌──────────────────────────────────────────────────────────┐
│                   AI API Gateway                         │
│                                                          │
│  User Request ("What is our Q3 revenue?")                │
│       │                                                  │
│       ▼                                                  │
│  ┌───────────────────────┐                               │
│  │ Semantic Cache Check  │ ──(Hit)──> Return Fast Answer │
│  │ (Redis + Embeddings)  │                               │
│  └────┬──────────────────┘                               │
│       │ (Miss)                                           │
│       ▼                                                  │
│  ┌───────────────────────┐                               │
│  │ RAG Retrieval         │ (Fetches 10k tokens of docs)  │
│  └────┬──────────────────┘                               │
│       │                                                  │
│       ▼                                                  │
│  ┌───────────────────────┐                               │
│  │ Context Compressor    │ (Shrinks 10k -> 2k tokens)    │
│  │ (LLMLingua API)       │                               │
│  └────┬──────────────────┘                               │
│       │                                                  │
│       ▼                                                  │
│  ┌───────────────────────┐                               │
│  │ Inference Engine      │ (vLLM / SGLang)               │
│  │ (Prefix Cache Hit)    │ Prefill skipped for system    │
│  │                       │ prompt. Generates answer.     │
│  └────┬──────────────────┘                               │
│       │                                                  │
│       ▼                                                  │
│  ┌───────────────────────┐                               │
│  │ Semantic Cache Store  │ (Saves Q&A for next time)     │
│  └───────────────────────┘                               │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Production Incident Scenarios

### Incident 1: "Semantic Cache Returning Wrong Answers"
**Symptoms:** User asks "Cancel my subscription". The system instantly replies "Your subscription to the Premium tier has been upgraded."
**Root Cause:** The semantic cache threshold was set too low (e.g., 0.70). The vector space considered "Cancel my subscription" and "Upgrade my subscription" to be highly semantically similar because they share many contextual features.
**Fix:** 
1. Increase the similarity threshold to 0.95.
2. Implement **Exact Match / Intent Caching** for high-risk actions. Do not use semantic caching for mutations (writes/updates), only for queries (reads).

### Incident 2: "Prefix Cache Eviction Thrashing"
**Symptoms:** Prompt caching hit rate drops to 5%. Latency and compute costs spike.
**Root Cause:** The inference cluster receives requests from 50 different enterprise tenants. Each tenant has a unique 2,000-token system prompt. The GPU VRAM can only hold 10 system prompts in the cache at a time. The cache continuously evicts Tenant A to make room for Tenant B, resulting in zero cache hits.
**Fix:** Implement **Tenant-Aware Routing** at the API Gateway. Route all requests for Tenant A to vLLM Replica 1. Route Tenant B to vLLM Replica 2. The prefix cache for each replica remains warm.

### Incident 3: "Cost Explodes Due to Chat History"
**Symptoms:** API costs increase exponentially as a conversation gets longer.
**Root Cause:** The application passes the entire conversation history (turn 1 to turn 50) in every API call. Because the prompt is constantly changing, Prompt Caching is defeated (hashes don't match).
**Fix:** 
1. Summarize old chat history. Pass a 100-token summary + the last 3 turns, instead of 50 turns.
2. Put the dynamic variables (chat history) at the *end* of the prompt, and the static variables (system instructions, huge rule sets) at the *beginning* of the prompt to maximize prefix caching.

---

## 6. Cost Optimization Economics (The Math)

If you process 1,000 requests per minute with a 10,000 token context window on Llama-3 70B (FP16):

**Naive Pipeline:**
- Tokens/min: 10,000,000
- GPUs needed: ~10x H100s
- Monthly Infrastructure Cost: ~$20,000

**Optimized Pipeline:**
1. **Semantic Cache (20% hit rate):** Reduces to 8,000 requests.
2. **Context Compression (5x):** Reduces context from 10k to 2k.
3. **Prefix Caching (1k system prompt):** Skips prefill for 1k tokens.
- New Tokens/min to process: 8,000 × 1,000 = 8,000,000 (an 80% reduction in prefill compute).
- GPUs needed: 2-3x H100s
- Monthly Infrastructure Cost: ~$5,000

---

## 7. Kubernetes / Implementation Perspective

### GPTCache Deployment

Use an open-source framework like **GPTCache** as a proxy layer in your Kubernetes cluster.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gptcache-proxy
spec:
  template:
    spec:
      containers:
      - name: gptcache
        image: zilliz/gptcache:latest
        env:
        - name: CACHE_STORE_URL
          value: "redis://redis-cluster:6379"
        - name: VECTOR_DB_URL
          value: "http://milvus-server:19530"
```

---

## Summary

```
Cost Optimization Checklist:

1. Put STATIC content at the TOP of your prompts (System rules, RAG context).
2. Put DYNAMIC content at the BOTTOM of your prompts (User questions).
   (This ensures Prefix Caching works).
3. Deploy a Semantic Cache (Redis/GPTCache) in front of your LLMs.
4. Compress large RAG contexts using LLMLingua before sending to expensive models.
5. Route requests from the same user/tenant to the same GPU node to maximize cache hits.
```

---
*Next: [13 — Observability using OpenTelemetry, Prometheus & Tracing →](13_Observability_Tracing_Deep_Dive.md)*
