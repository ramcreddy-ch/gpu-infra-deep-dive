# 02. GPU AI Workloads & Utilization (10/10 Enterprise Depth)

AI is essentially massive arrays (Tensors) being multiplied. But understanding how those Tensors map into the 80 gigabytes of an NVIDIA A100 dictates whether your cluster succeeds or crashes with `CUDA Out of Memory`.

---

## 1. The vRAM Heavyweights: A Mathematical Breakdown

Let's do the actual production math for loading a **LLaMA-3 70B** parameter model into memory for training.

### Parameter Storage Format
Parameters (Weights) are stored in varying precisions. 
*   **FP32 (Single Precision):** 4 Bytes per parameter. (Rarely used for large models).
*   **BF16/FP16 (Half Precision):** 2 Bytes per parameter. (Standard for Training).
*   **INT8/INT4 (Quantized):** 1 Byte or 0.5 Bytes per parameter. (Inference).

**Model Weights size (BF16):** 
70 Billion Parameters * 2 Bytes = **~140 GB**. 
*(Immediately, we see a LLaMA 70B model physically cannot fit on a single 80GB GPU. We must shard it across multiple GPUs).*

### Training Memory Constituents
If we train this model using Adam Optimizer, VRAM explodes due to multiple active states:
1.  **Model Weights:** 140 GB
2.  **Gradients:** Needed for the backward pass. Equals model size. (+ 140 GB).
3.  **Optimizer States:** Adam stores momentum and variance for *every* parameter in FP32 (4 bytes + 4 bytes = 8 bytes). 
    *   70 Billion * 8 Bytes = (+ 560 GB).
4.  **Activations:** Intermediate layer outputs saved during the forward pass. This scales linearly with Batch Size and Sequence Length. (Roughly ~20-50 GB depending on context length).

**Total Theoretical Memory strictly for Training LLaMA 70B:** ~850 GB of VRAM. 
*To train this, we need an HGX chassis of 8x H100 (80GB) GPUs (640GB total VRAM), plus DeepSpeed ZeRO Offloading or highly sharded Tensor Parallelism across multiple nodes.*

---

## 2. Real-World Profiling: Moving Beyond `nvidia-smi`

`nvidia-smi` is a liar. It reports `GPU-Util: 100%` if simply reading a file from disk locks the GPU context queue, even if 99% of your CUDA cores are physically powered down. 

To achieve 10/10 performance engineering, you must analyze the timeline.

### NVIDIA Nsight Systems (`nsys`)
Nsight Systems gives you the absolute truth about API execution timelines. Wrap your PyTorch script with `nsys`:

```bash
nsys profile --trace=cuda,cudnn,cublas,osrt -o my_training_profile python train.py
```
This generates a `.qdrep` file. When you open this in the Nsight GUI, you look for **White Space**.
*   **Kernel Spacing:** If you see gaps of white space between CUDA Kernels on the timeline, your CPU is failing to queue work fast enough. This usually means your Dataloaders are too slow (I/O bound reading PNGs/JPEGs), or python GIL lock is interfering.
*   **Solution:** Increase `num_workers` in PyTorch's DataLoader or implement NVIDIA DALI (Data Loading Library) to move image/text decoding directly to the GPU hardware decoders.

### PyTorch Memory Profiling
To understand exactly which layers cause an OOM:
```python
import torch

# Start recording memory history
torch.cuda.memory._record_memory_history()

# Execute your heavy model logic
model(inputs) 
loss.backward()

# Dump the snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```
Upload that snapshot to `https://pytorch.org/memory_viz`. The visualization shows exactly when PyTorch's caching allocator fragments memory. If you have "Free" memory but still OOM, it's due to **Fragmentation** (the contiguous blocks of free space are smaller than the incoming tensor).

---

## 3. Inference Specifics: The KV Cache Wall

Inference is memory-bandwidth bound due to autoregressive decoding (predicting token-by-token).
*   To avoid re-computing attention for all past tokens, the model saves the `Keys` and `Values` of every generated token in the **KV Cache**.

**KV Cache Math:**
$$\text{Memory Allocation} = 2 (\text{K and V}) \times 2 (\text{FP16 bytes}) \times \text{Number of Layers} \times \text{Hidden Size} \times \text{Sequence Length} \times \text{Batch Size}$$

If you serve an LLM with a 100,000 token context window, the KV cache footprint will rapidly surpass the size of the Model Weights themselves, crashing your inference endpoint. This is what gave birth to advanced frameworks like **vLLM** and **PagedAttention**.
