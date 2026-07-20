# MLOps, MLflow, Model Registry, CI/CD, GitOps & AI Platform Automation

> *Data scientists build models in notebooks. Platform engineers build pipelines that reliably deploy those models 1,000 times a day. MLOps is the engineering discipline of treating AI models as software artifacts with versions, tests, and automated deployment pipelines.*

---

## 1. What Problem Does This Solve?

### The "Notebook Toss" Anti-Pattern

In immature organizations, the workflow looks like this:
1. Data Scientist trains a model in a Jupyter Notebook on their laptop.
2. They save the model as `model_final_v3_really_final.pt`.
3. They Slack the file to the backend engineer.
4. The backend engineer wraps it in Flask and deploys it.
5. Three weeks later, the model starts predicting garbage. No one knows what data it was trained on, what hyper-parameters were used, or how to reproduce the build.

**The Solution:** MLOps brings standard software engineering practices (GitOps, CI/CD, Versioning) to Machine Learning.

---

## 2. Internal Architecture

### The Enterprise MLOps Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Enterprise MLOps Pipeline                        │
│                                                                         │
│  1. Code & Config (Git / GitHub)                                        │
│     ├── train.py                                                        │
│     └── config.yaml (Learning rate, batch size)                         │
│                                                                         │
│  2. Data Layer (Feature Store / S3)                                     │
│     └── Versioned datasets (DVC or Delta Lake)                          │
│                                                                         │
│  3. Training Compute (Kubernetes / Ray)                                 │
│     └── Executes train.py on GPUs                                       │
│                                                                         │
│  4. Experiment Tracking & Model Registry (MLflow)                       │
│     ├── Logs metrics (Loss, Accuracy)                                   │
│     ├── Logs artifacts (model.safetensors)                              │
│     └── Registers Model Version (e.g., Llama-FineTune-v1.2.0)           │
│                                                                         │
│  5. Continuous Deployment (ArgoCD / KServe)                             │
│     └── Pulls model from Registry, deploys to Triton/vLLM               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Internal Working

### Experiment Tracking vs. Model Registry

These two concepts are often confused, but they serve different phases of the lifecycle.

**1. Experiment Tracking (The Lab):**
When a data scientist runs 50 different training loops to find the best hyper-parameters, MLflow logs every run.
- Run 1: LR=1e-4, Loss=0.5
- Run 2: LR=1e-5, Loss=0.2 (Best)
*This is for R&D.*

**2. Model Registry (The Factory):**
When the data scientist decides Run 2 is good enough for production, they "register" it.
The Model Registry acts like a Docker Container Registry (e.g., Artifactory). It holds immutable, versioned model artifacts ready for deployment.
- State: `Staging` -> `Production` -> `Archived`

### The Feature Store

In traditional ML, a model might require the feature "User's average spend over 30 days".
- The Data Scientist writes a slow SQL query to calculate this during training.
- The Backend Engineer writes a fast Java service to calculate this during real-time inference.
- **The Bug:** The SQL and Java logic are slightly different. The model fails in production (Training-Serving Skew).

**Feature Store (e.g., Feast, Hopsworks):** A centralized database where features are defined once.
- The training job pulls historical features (Offline Store - Parquet/S3).
- The inference API pulls real-time features (Online Store - Redis) using the exact same definition.

---

## 4. Production Architecture

### LLMOps: Continuous Fine-Tuning Pipeline (GitOps)

For LLMs, models degrade as language and facts change. We need an automated pipeline to continuously fine-tune the model on new data.

```
┌──────────────────────────────────────────────────────────────────┐
│                  Automated LLM Training CI/CD                    │
│                                                                  │
│  ┌────────────┐                                                  │
│  │ Git Push   │ (Developer updates training script/config)       │
│  └──────┬─────┘                                                  │
│         │                                                        │
│         ▼                                                        │
│  ┌───────────────────────────┐                                   │
│  │ GitHub Actions (CI)       │                                   │
│  │ 1. Lints code             │                                   │
│  │ 2. Runs unit tests        │                                   │
│  │ 3. Triggers K8s Job       │                                   │
│  └──────┬────────────────────┘                                   │
│         │                                                        │
│         ▼                                                        │
│  ┌───────────────────────────┐      ┌─────────────────────────┐  │
│  │ Kubeflow Training Job     │─────>│ MLflow Tracking Server  │  │
│  │ (Runs distributed LoRA)   │      │ (Logs loss, saves       │  │
│  └──────┬────────────────────┘      │  safetensors artifact)  │  │
│         │                           └─────────────────────────┘  │
│         ▼                                                        │
│  ┌───────────────────────────┐                                   │
│  │ Automated Evaluation      │                                   │
│  │ (Runs LLM-as-a-judge on   │                                   │
│  │  the new artifact)        │                                   │
│  └──────┬────────────────────┘                                   │
│         │ If Pass (Loss < x, Eval > y)                           │
│         ▼                                                        │
│  ┌───────────────────────────┐                                   │
│  │ Update Git Manifest       │                                   │
│  │ (Updates KServe YAML to   │                                   │
│  │  point to new S3 URI)     │                                   │
│  └──────┬────────────────────┘                                   │
│         │                                                        │
│         ▼                                                        │
│  ┌───────────────────────────┐                                   │
│  │ ArgoCD (GitOps CD)        │                                   │
│  │ Detects YAML change,      │                                   │
│  │ performs Canary rollout   │                                   │
│  │ to vLLM pods.             │                                   │
│  └───────────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Production Incident Scenarios

### Incident 1: "The Silent Model Rollback"
**Symptoms:** An automated CI/CD pipeline deployed a new model version. ArgoCD showed it as "Healthy". However, the application started failing.
**Root Cause:** The KServe predictor pod downloaded the model from S3. The new model was 140GB. The node only had a 100GB disk. The container crashed with `DiskQuotaExceeded`. ArgoCD automatically rolled back to the old version (which was still on disk) to maintain availability. No one noticed the deployment failed until weeks later.
**Fix:** 
1. Always size ephemeral storage requests (`ephemeral-storage: "200Gi"`) correctly in Kubernetes.
2. Implement specific Prometheus alerts for `kserve_model_load_failed`.

### Incident 2: "The Pickled RCE"
**Symptoms:** A data scientist downloaded an open-source model from Hugging Face and uploaded it to the internal MLflow registry. When the KServe pod loaded the model, it executed a malicious bash script that compromised the K8s cluster.
**Root Cause:** The model was saved using Python's `pickle` format (standard PyTorch `.pt` or `.bin`). Pickle files are executable code, not just data arrays. Loading a malicious pickle file results in Remote Code Execution (RCE).
**Fix:** 
1. **Never use Pickle in production.** Always enforce the use of `.safetensors` (which only contains data, no code).
2. Use tools like `ModelScan` in your CI/CD pipeline to scan all artifacts for serialized malware before allowing them into the Model Registry.

### Incident 3: "Training-Serving Feature Skew"
**Symptoms:** A fraud detection model had 99% accuracy during training (logged in MLflow). In production, its accuracy dropped to 40%.
**Root Cause:** The data scientists normalized the "transaction amount" feature by dividing by the *global maximum* amount found in the historical training dataset. In production, the backend engineer normalized it by dividing by 10,000 (a hardcoded guess). The model was receiving totally different numerical scales.
**Fix:** Implement a Feature Store (e.g., Feast) to ensure the exact same normalization logic is applied to both offline and online data.

---

## 6. Performance Optimization

### Optimize the Model Download Phase

When scaling an inference endpoint from 1 to 50 pods, each pod must download the 140GB model from S3.
- 50 pods × 140GB = 7 Terabytes of network traffic.
- At 10 Gbps, this takes ~2 hours. The autoscaler is useless.

**Solutions:**
1. **Host Caching (NVIDIA/K8s Dataset PVCs):** Mount an NVMe drive to the K8s node. Download the model *once* to the node's NVMe drive. Use a `hostPath` mount to share it with all pods on that node. Pod startup drops from 5 minutes to 5 seconds.
2. **S3 Gateway Endpoints:** Ensure S3 traffic does not traverse the NAT Gateway, which is incredibly slow and expensive. Use VPC Endpoints.
3. **BitTorrent for Models:** Advanced organizations (like Meta) use peer-to-peer protocols to distribute massive model weights across the cluster, rather than overwhelming a single storage server.

---

## Summary

```
MLOps Maturity Levels:
Level 0: Manual notebook training, scp models to servers. (Danger)
Level 1: Automated training pipelines, MLflow tracking. (Good)
Level 2: Automated CI/CD, Model Registry, GitOps deployments. (Better)
Level 3: Automated CT (Continuous Training) triggered by data drift. (Best)

Golden Rule: The model artifact is just the output. The true product is the pipeline that created the artifact.
```

---
*Next: [20 — Linux Internals, Storage & OS Optimization for AI →](20_Linux_Internals_OS_Optimization_Deep_Dive.md)*
