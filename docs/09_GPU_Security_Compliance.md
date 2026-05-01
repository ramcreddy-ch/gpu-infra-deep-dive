# 09. GPU Security, Compliance, and Multi-Tenancy

Most GPU security guides are an afterthought — a paragraph at the end of a tutorial saying "remember to set RBAC." In production, GPU security is genuinely harder than CPU security because GPUs bypass many of the isolation primitives that Linux and Kubernetes rely on. Here's what I've learned the hard way.

**Author:** [Ramchandra Chintala](https://github.com/ramcreddy-ch)

---

## Why GPU Security Is Different

When you run a process on a CPU inside a Kubernetes container, Linux cgroups and namespaces provide strong isolation. The container cannot see other containers' memory or processes.

GPUs don't play by these rules:
- **GPU memory is not namespace-isolated.** Without MIG, two containers sharing a time-sliced GPU can potentially snoop on each other's vRAM.
- **CUDA drivers operate in kernel space.** A GPU driver bug or exploit can escalate to root on the host.
- **Model weights are intellectual property.** If someone extracts your fine-tuned LLM weights from GPU memory, that's a major IP breach.

---

## Multi-Tenancy: Sharing GPUs Without Sharing Secrets

### The Problem

Your ML Research team and the Production Inference team both need GPUs. But they should never be able to access each other's data, models, or GPU memory.

### Option 1: Physical Isolation (Separate Node Pools)

The simplest and most secure approach. Each team gets their own GPU nodes.

```yaml
# Team A: ML Research (can use expensive A100s)
nodeGroups:
  - name: gpu-research
    labels:
      team: ml-research
      workload-type: gpu-training
    taints:
      - key: team
        value: ml-research
        effect: NoSchedule

# Team B: Production Serving (cost-optimized T4/L4)
  - name: gpu-serving
    labels:
      team: production
      workload-type: gpu-serving
    taints:
      - key: team
        value: production
        effect: NoSchedule
```

**Pros:** Zero cross-contamination risk. Simple.
**Cons:** Expensive. GPUs sit idle when one team isn't using them.

### Option 2: MIG Isolation (Hardware-Level Sharing)

If you must share a physical GPU between tenants, MIG is the only option I'd trust in production. Time-slicing has no memory isolation.

```bash
# Enable MIG on an A100
sudo nvidia-smi -i 0 -mig 1

# Create isolated instances
sudo nvidia-smi mig -i 0 -cgi 9,9,9,9,9,9,9 -C
# Creates 7x 1g.10gb instances, each with isolated memory and compute
```

Each MIG instance gets its own device file (`/dev/nvidia0`, `/dev/nvidia1`, etc.) and appears as a separate GPU to Kubernetes. There is zero memory cross-talk at the hardware level.

### Option 3: Network Policies for GPU Namespaces

Even with physical node isolation, you need network policies to prevent pods from reaching each other's services:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-cross-team-traffic
  namespace: ml-research
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              team: ml-research
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              team: ml-research
    - to: []  # Allow DNS
      ports:
        - port: 53
          protocol: UDP
```

---

## Protecting Model Weights

Your fine-tuned models are potentially worth millions in training compute. Treat them like source code.

### 1. Encrypted Storage at Rest
Model artifacts in S3/GCS/ADLS must use server-side encryption with customer-managed keys (CMK):

```hcl
# Terraform: S3 bucket for model artifacts
resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.model_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.ml_platform.arn
    }
  }
}
```

### 2. Pod Security Standards
Prevent containers from running as root or escalating privileges:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: trainer
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      # GPU workloads still work with these restrictions
      # because the NVIDIA container runtime handles device injection
```

### 3. Audit Logging for GPU Access
Track who ran what on which GPU:

```yaml
# Falco rule for GPU access monitoring
- rule: Unauthorized GPU Access
  desc: Detects processes accessing NVIDIA GPU devices unexpectedly
  condition: >
    open_write and
    fd.name startswith /dev/nvidia and
    not container.image.repository in (approved_gpu_images)
  output: >
    Unauthorized GPU device access
    (user=%user.name command=%proc.cmdline container=%container.name
     image=%container.image.repository gpu_device=%fd.name)
  priority: WARNING
```

---

## RBAC: Who Can Deploy GPU Workloads?

Not everyone should be able to request GPUs. I set up a tiered RBAC system:

```yaml
# Role: Can submit GPU training jobs (Data Scientists)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gpu-user
  namespace: ml-training
rules:
  - apiGroups: ["kubeflow.org"]
    resources: ["pytorchjobs", "mpijobs"]
    verbs: ["create", "get", "list", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["pods/exec"]
    verbs: []  # No exec into GPU pods!

---
# Role: Can manage GPU infrastructure (Platform Engineers)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: gpu-admin
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "patch"]  # Can taint/label nodes
  - apiGroups: ["nvidia.com"]
    resources: ["*"]
    verbs: ["*"]
```

**Why no `pods/exec`?** If a Data Scientist can `exec` into a GPU pod, they can run `nvidia-smi` and potentially see memory contents. In a multi-tenant environment, this is an information leak.

---

## Compliance Considerations

### SOC 2 / HIPAA
- All GPU nodes must be in private subnets (no public IPs).
- Model training data containing PII must be encrypted in transit (TLS 1.3) and at rest.
- GPU node access must be audited via CloudTrail/Activity Log.

### Cost Governance
- Set `ResourceQuota` per namespace to cap GPU usage per team.
- Use `LimitRange` to prevent a single pod from requesting all GPUs on a node.

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: ml-training
spec:
  limits:
    - type: Container
      max:
        nvidia.com/gpu: "4"  # No single container can request more than 4 GPUs
      default:
        nvidia.com/gpu: "1"
```

---

*Next: [10. GPU Monitoring Deep Dive & Prometheus Metrics](./10_GPU_Monitoring_Prometheus.md)*
