# LLM Evaluation, Benchmarks, Hallucination Detection & Model Quality

> *In traditional software engineering, you write unit tests (`assert result == 5`). In AI infrastructure, the output is non-deterministic, creative text. How do you "unit test" poetry? How do you automatically detect if your new RAG pipeline is lying to customers? This chapter covers the rigorous, automated evaluation of LLM quality.*

---

## 1. What Problem Does This Solve?

### The "Vibes-Based" Deployment Anti-Pattern

Many AI teams update their LLM system (e.g., swapping Llama-3-8B for Mistral-7B, or changing the system prompt), run 5 manual queries in a UI, decide "it looks good," and deploy it to production.

This is known as **Vibes-Based Evaluation**. It is dangerous.
- The new model might be great at coding but suddenly terrible at Spanish.
- The new system prompt might make the model slightly more aggressive over a 10-turn conversation.
- A quantization change (FP16 to INT4) might cause catastrophic failure on edge-case math problems.

**The Solution:** Automated, programmatic evaluation pipelines (EvalOps) that score models on thousands of test cases before deployment.

---

## 2. Internal Architecture

### The EvalOps Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Evaluation Pipeline                │
│                                                             │
│  1. Pull Request created (e.g., changed RAG system prompt)  │
│                           │                                 │
│                           ▼                                 │
│  2. Generate Outputs (Dataset: 1,000 queries)               │
│     Target Model generates 1,000 answers.                   │
│                           │                                 │
│                           ▼                                 │
│  3. LLM-as-a-Judge Evaluation                               │
│     GPT-4o or Claude-3.5-Sonnet scores the 1,000 answers    │
│     against the Golden Dataset.                             │
│                           │                                 │
│                           ▼                                 │
│  4. Metrics Aggregation (Ragas / TruLens / Phoenix)         │
│     Calculates: Context Precision, Context Recall,          │
│     Faithfulness, Answer Relevance.                         │
│                           │                                 │
│                           ▼                                 │
│  5. CI/CD Gate                                              │
│     If (Faithfulness < 0.95) -> FAIL PR                     │
│     If (Pass) -> Merge and Deploy                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Internal Working

### LLM-as-a-Judge

Using humans to evaluate 1,000 answers is too slow and expensive.
Using string matching (BLEU/ROUGE) is terrible for LLMs (e.g., "The cat sat" and "A feline rested" have 0% BLEU overlap but mean the exact same thing).

**The Industry Standard:** Use a frontier model (GPT-4) to evaluate the outputs of a cheaper/smaller model.

```text
# Example Prompt given to the Judge LLM
You are an expert evaluator.
Given the User Question, the Retrieved Context, and the AI Answer, 
score the AI Answer on "Faithfulness" from 1 to 5.
Faithfulness means the Answer does not hallucinate facts outside the Context.

Question: {user_question}
Context: {rag_context}
Answer: {ai_answer}

Provide your score and a 1-sentence reasoning in JSON format.
```

### The RAG Triad Metrics (e.g., TruLens / Ragas)

Evaluating RAG requires breaking the pipeline into three distinct measurements:

1. **Context Relevance (Retrieval Quality):** Did the Vector DB return documents that actually answer the user's question? (If this fails, the Vector DB / Embeddings are the problem).
2. **Faithfulness / Groundedness (Hallucination Detection):** Did the LLM base its entire answer *only* on the retrieved context? (If this fails, the LLM hallucinated).
3. **Answer Relevance (End-to-End Quality):** Does the final answer actually address the user's original question? (If this fails, the LLM went off-topic).

---

## 4. Production Architecture

### Continuous Evaluation (Shadow Mode)

In production, user queries are unpredictable. You cannot rely solely on offline datasets. You must run continuous evaluation on live traffic.

```
┌─────────────────────────────────────────────────────────────┐
│                 Live Traffic Evaluation                     │
│                                                             │
│  Live User Request ──> API Gateway ──> Target Model (Llama) │
│                               │                             │
│                      (Async Shadow Copy)                    │
│                               │                             │
│                               ▼                             │
│                       Kafka / PubSub                        │
│                               │                             │
│                               ▼                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Evaluation Workers (Celery / Ray)                     │  │
│  │ Select 5% of live traffic.                            │  │
│  │ Run LLM-as-a-Judge to score Faithfulness and Toxicity.│  │
│  │ Log scores to Prometheus / Grafana.                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Production Incident Scenarios

### Incident 1: "The Quantization Quality Drop"
**Symptoms:** Engineering deployed a GPTQ INT4 quantized model to reduce inference costs. Latency was great, but customer satisfaction dropped by 20%.
**Root Cause:** The team did not run the MMLU or HumanEval benchmarks after quantizing. While the model could still speak English perfectly, the INT4 quantization destroyed its ability to perform basic arithmetic and follow multi-step reasoning constraints.
**Fix:** Implement a rigorous automated benchmark suite (using `lm-evaluation-harness`) that runs before any quantized model is promoted to production.

### Incident 2: "The Re-Ranker Hallucination Trap"
**Symptoms:** The LLM started confidently inventing internal company policies.
**Root Cause:** The RAG evaluation dashboard showed **Faithfulness = 98%** but **Context Relevance = 12%**. The Vector DB was retrieving totally irrelevant documents. The LLM was correctly reading those irrelevant documents, realizing they didn't contain the answer, and relying on its base training weights to guess the company policy.
**Fix:** 
1. Improve the retrieval pipeline (implement a Cross-Encoder Re-ranker).
2. Update the system prompt: *"If the provided context does not contain the answer, you must reply 'I do not know'. Do not guess."*

### Incident 3: "A/B Test Route Flapping"
**Symptoms:** Users reported the chatbot's personality and capability changed wildly between messages in the same conversation.
**Root Cause:** An A/B test was routing 50% of traffic to Model A and 50% to Model B. However, the routing was done *per API request*, not *per user session*. Message 1 went to GPT-4, Message 2 went to Llama-3-8B.
**Fix:** Implement sticky sessions (cookie-based or user-ID hash routing) at the AI Gateway to ensure a single conversation stays on the same model.

---

## 6. Performance Optimization

### Optimizing the Judge

Running GPT-4 as a judge on 10,000 evaluation rows is slow and costs $200.
**Optimization:** Use **Judge Distillation**.
1. Use GPT-4 to score 1,000 examples.
2. Fine-tune a tiny, cheap model (like Llama-3-8B) on those 1,000 scored examples to act as the judge.
3. Deploy the cheap judge locally via vLLM. It will score the remaining 9,000 rows in seconds for $0.

---

## 7. Open Source Tooling Landscape

- **EleutherAI LM Eval Harness:** The industry standard for running public benchmarks (MMLU, GSM8k, HellaSwag).
- **Ragas (Retrieval Augmented Generation Assessment):** Framework specifically for evaluating RAG pipelines without requiring human-labeled ground truth.
- **TruLens:** Provides the "RAG Triad" feedback functions.
- **Arize Phoenix:** Excellent UI for visualizing traces, embeddings (UMAP projection), and evaluation scores.

---

## Summary

```
Evaluation Checklist for Production Release:
1. Public Benchmarks (MMLU, GSM8k) to ensure baseline competence hasn't degraded.
2. Golden Dataset (100-500 custom company queries) evaluated by LLM-as-a-Judge.
3. RAG Triad scores (Context Relevance, Faithfulness, Answer Relevance) > 0.90.
4. Toxicity and Prompt Injection vulnerability scans.
5. Latency and Throughput load tests (Locust / K6).
```

---
*Next: [15 — Model Routing, Intelligent Fallback & Gateway Architectures →](15_Model_Routing_Gateway_Deep_Dive.md)*
