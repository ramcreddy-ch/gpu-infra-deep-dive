# Embeddings, Vector Databases, Hybrid Search & Enterprise RAG Pipelines

> *Retrieval-Augmented Generation (RAG) is how you give LLMs long-term memory and access to proprietary data. While the concept is simple, building a RAG pipeline that achieves >95% retrieval accuracy at sub-100ms latencies across billions of documents requires advanced distributed systems engineering.*

---

## 1. What Problem Does This Solve?

### The Hallucination & Knowledge Cutoff Problem

Large Language Models (LLMs) suffer from three fatal flaws for enterprise use cases:
1. **Knowledge Cutoff:** They only know what they were trained on up to a certain date.
2. **Proprietary Ignorance:** They don't know your company's internal Slack messages, Jira tickets, or private source code.
3. **Hallucinations:** When asked a question they don't know, they confidently invent plausible-sounding lies.

### The Naive Solution (Fine-Tuning) vs. The Real Solution (RAG)

Many enterprises assume they must fine-tune an LLM on their proprietary data. **This is usually a mistake.**
Fine-tuning teaches the model *style* and *format*, but it is terrible at teaching the model *facts*.

**Retrieval-Augmented Generation (RAG)** solves this by giving the LLM a search engine:
1. User asks: "What is the error code 504 on the payment gateway?"
2. System searches internal docs for "error 504 payment gateway".
3. System injects the top 5 search results into the LLM's system prompt.
4. LLM reads the context and generates an accurate, cited answer.

---

## 2. Internal Architecture

### The Enterprise RAG Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Data Ingestion (Offline)                      │
│                                                                         │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌─────────────────┐   │
│  │ Confluence│─>│ Document   │─> │ Chunker    │─> │ Embedding Model │   │
│  │ S3 / Jira │  │ Parsers    │   │ (LangChain)│   │ (GPU Server)    │   │
│  └──────────┘   │ (Unstructured) │            │   │ e.g. BGE-Large  │   │
│                 └────────────┘   └────────────┘   └────────┬────────┘   │
│                                                            │            │
│                                                            ▼            │
│                                                   ┌─────────────────┐   │
│                                                   │ Vector Database │   │
│                                                   │ (Milvus / Qdrant│   │
│                                                   └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           Query Execution (Online)                      │
│                                                                         │
│  ┌────────┐  ┌─────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │ User   │─>│ Query       │─>│ Embedding Model │─>│ Vector DB Search│   │
│  │ Prompt │  │ Rewriter    │  │ (GPU Server)    │  │ (ANN HNSW)     │   │
│  └────────┘  │ (LLM router)│  └─────────────────┘  └──────┬─────────┘   │
│              └─────────────┘                              │             │
│                                                           ▼             │
│  ┌────────┐  ┌─────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │ Answer │<─│ Generation  │<─│ Prompt Builder  │<─│ Re-Ranker      │   │
│  │ Stream │  │ LLM (vLLM)  │  │ (Inject context)│  │ (Cross-Encoder)│   │
│  └────────┘  └─────────────┘  └─────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Internal Working

### Embeddings & Vector Space

An embedding model (like `bge-large-en-v1.5`) takes a string of text and outputs a dense vector of floating-point numbers (e.g., a 1024-dimensional array).

```python
vector = [0.012, -0.453, 0.892, ..., 0.034] # Length 1024
```

In this 1024-dimensional space, sentences with similar **semantic meaning** are grouped close together, regardless of exact keyword matches.
"The feline rested on the rug" is mathematically very close to "The cat sat on the mat."

### Vector Search Internals (HNSW)

Finding the closest vector among 1 billion vectors using brute force (calculating Cosine Similarity for every pair) is $O(N)$. It is too slow for real-time.

Vector databases use **Approximate Nearest Neighbor (ANN)** algorithms. The most dominant is **HNSW (Hierarchical Navigable Small World)**.

**How HNSW works:**
It builds a multi-layered graph.
1. Top layer: Very few nodes (fast to traverse, gives general direction).
2. Middle layers: More nodes (narrows down the neighborhood).
3. Bottom layer: All vectors (finds the exact neighbors).

*Analogy:* To find a specific address in Tokyo, you first fly to Japan (Top layer), then take a train to Tokyo (Middle layer), then walk to the specific street (Bottom layer). HNSW achieves $O(\log N)$ search time.

### Hybrid Search & Reciprocal Rank Fusion (RRF)

Vector search is great for semantics, but terrible for specific keywords, names, or IDs. (e.g., searching for "Error Code 409X").
Standard Keyword Search (BM25 / Elasticsearch) is great for keywords, but terrible for semantics.

**Hybrid Search** runs both simultaneously and merges the results.

```python
# Reciprocal Rank Fusion (RRF) Algorithm
# Merges results from Vector Search and Keyword Search

def rrf(vector_results, keyword_results, k=60):
    rrf_scores = {}
    
    # Process vector ranks
    for rank, doc in enumerate(vector_results):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + (1.0 / (k + rank))
        
    # Process keyword ranks
    for rank, doc in enumerate(keyword_results):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + (1.0 / (k + rank))
        
    # Sort by combined score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 4. Production Architecture

### Enterprise Vector DB Cluster (Milvus/Qdrant on K8s)

```
┌──────────────────────────────────────────────────────────┐
│              Distributed Vector Database                 │
│                                                          │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐        │
│  │ Query Node │   │ Query Node │   │ Query Node │        │
│  │ (CPU only) │   │ (CPU only) │   │ (CPU only) │        │
│  └──────┬─────┘   └──────┬─────┘   └──────┬─────┘        │
│         │                │                │              │
│  ┌──────▼────────────────▼────────────────▼──────┐       │
│  │                  Proxy Layer                  │       │
│  └──────┬────────────────┬────────────────┬──────┘       │
│         │                │                │              │
│  ┌──────▼─────┐   ┌──────▼─────┐   ┌──────▼─────┐        │
│  │ Data Node  │   │ Data Node  │   │ Data Node  │        │
│  │ Shard 0    │   │ Shard 1    │   │ Shard 2    │        │
│  │ RAM: 128GB │   │ RAM: 128GB │   │ RAM: 128GB │        │
│  └────────────┘   └────────────┘   └────────────┘        │
│                                                          │
│  Persistent Storage: MinIO / S3 (For segment backups)    │
│  Metadata: etcd                                          │
│  Streaming Log: Apache Pulsar / Kafka                    │
└──────────────────────────────────────────────────────────┘
```

**Why CPU for Vector DBs?**
Most vector search (HNSW traversal) requires heavy memory random access. GPUs are terrible at random access (they excel at linear matrix multiplication). Therefore, Vector DB query nodes are usually massive CPU instances with hundreds of gigabytes of RAM. The *Embedding Generation* happens on the GPU, but the *Vector Search* happens on the CPU.

---

## 5. Production Incident Scenarios

### Incident 1: "Vector DB OOMKilled under Load"
**Symptoms:** The Qdrant/Milvus data nodes crash with OOMKilled when the index reaches 50 million documents.
**Root Cause:** HNSW graphs must reside in RAM for fast traversal. The vectors plus the graph edges consume massive memory. 50M embeddings (1024-dim, FP32) = 200 GB for vectors + ~50 GB for the HNSW graph. The node only had 128GB RAM.
**Fix:** 
1. Scale out (add more shards).
2. Enable **Scalar Quantization (SQ8)** or **Product Quantization (PQ)**. PQ compresses the vectors from FP32 to INT8 or smaller, reducing RAM usage by 4-8x at a slight cost to recall accuracy.

### Incident 2: "RAG Returning Irrelevant Information"
**Symptoms:** The LLM's answers are hallucinated or wrong, despite the data existing in the database.
**Investigation:** Evaluated the retrieval step independently. The Vector DB was returning document chunks that were too small (e.g., 50 tokens). The chunks contained pronouns ("He did it") but lacked the surrounding context to define "He".
**Fix:** Implement **Parent-Child Document Retrieval**.
- Chunk the document into small pieces (Child) for *searching* (better matching).
- When a Child matches, retrieve its large Parent chunk to feed into the LLM (better context).

### Incident 3: "Re-Ranker Bottlenecking the System"
**Symptoms:** RAG query latency spikes to 2.5 seconds.
**Root Cause:** The pipeline retrieves the top 100 documents and passes them to a Cross-Encoder Re-Ranker. The Re-Ranker model (running on CPU) takes 2 seconds to score 100 pairs of (query, document).
**Fix:** Deploy the Re-Ranker model to a GPU using Triton Inference Server with dynamic batching. Re-ranking is a heavy transformer forward pass and must be hardware-accelerated.

---

## 6. Performance Optimization

### GPU-Accelerated Embedding Generation

Embedding models (like `BGE` or `E5`) are usually small BERT-based models (~300M parameters). Do not serve them with naive PyTorch.

1. **Compile to TensorRT:** Since BERT models are static (encoder-only, no autoregressive generation), TensorRT provides massive speedups.
2. **Serve with Triton:** Use Triton's dynamic batching to process 256 embedding requests simultaneously.

### Vector DB Index Tuning

- **`m` parameter (HNSW):** Maximum number of connections per node. Default ~16. Higher `m` = better recall, slower build time, more RAM.
- **`ef_construction`:** Size of the dynamic list during index build. Default ~100. Higher = better index quality, much slower build.
- **`ef_search`:** Size of the dynamic list during query. Higher = better recall, slower search.

*Tuning strategy:* Build with high `m` and `ef_construction` (done offline). Tune `ef_search` at runtime based on the SLA vs Recall trade-off.

---

## 7. Security Perspective

### Document Level RBAC in RAG

When a user asks a question, the Vector DB must NOT return documents they aren't authorized to see.

**Anti-Pattern:** Retrieve top 100 docs, filter out unauthorized ones in Python.
*Why it fails:* If all 100 docs are unauthorized, the user gets 0 results, even if the 101st doc was authorized and highly relevant.

**Correct Architecture (Pre-Filtering):**
Vector databases support metadata filtering during the HNSW traversal.
```python
# User Alice belongs to groups: ['engineering', 'public']
client.search(
    collection_name="enterprise_docs",
    query_vector=embedded_query,
    # The Vector DB filters the graph dynamically during search
    query_filter=Filter(
        must=[
            FieldCondition(key="allowed_groups", match=MatchAny(any=["engineering", "public"]))
        ]
    )
)
```

---

## Summary

```
RAG Architecture Checklist:

1. Ingestion:
   - Use Parent-Child chunking.
   - Embed using a GPU-accelerated server (Triton).
   - Use Product Quantization on the Vector DB if scaling > 10M vectors.

2. Retrieval:
   - Use Hybrid Search (Vector + BM25 keyword).
   - Merge results using Reciprocal Rank Fusion (RRF).
   - Pre-filter results based on User RBAC permissions.

3. Generation:
   - Pass top 5-10 chunks through a Cross-Encoder Re-Ranker (GPU).
   - Inject top 3 chunks into the LLM system prompt.
   - Stream response to user.
```

---
*Next: [12 — Prompt Caching, Semantic Caching & AI Cost Optimization →](12_Prompt_Caching_Cost_Optimization_Deep_Dive.md)*
