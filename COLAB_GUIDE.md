# 🧭 Google Colab Evaluation Guide & Project Operations Manual

> **Purpose of this Document**:  
> Designed specifically for the **Google Colab runtime environment** and the **Colab Gemini Assistant**.  
> When switching to Google Colab, this manual provides immediate context on **system architecture, benchmark baselines, SOPs, and CLI execution commands**.

---

## 🎯 1. Project Positioning & System Architecture

This repository focuses on **Edge LLM Post-Training Quantization (PTQ)**, **numerical sensitivity analysis**, and **deep activation outlier topology mapping**.

### Core 5-Module Decoupled Architecture
The framework decouples quantization benchmarking into five standalone modules governed by an automated matrix controller:
1. **Module 1 (`src/data/loader.py`)**: Multi-domain token streaming and packing buffer (WikiText-2, C4, HumanEval, GSM8K; uniform 2048-token context windows; supports $N \in \{16, 128, 512\}$). Global parameters configured in `configs/pipeline_config.py`.
2. **Module 2 (`src/quantization/calibration.py` & `profiler.py`)**: Unified calibration engine based on `LayerObserver`, tracking extrema, 99.99% quantile clipping, channel-wise maximums, and symmetric/asymmetric scales (persisted to `.pt` and `.json`).
3. **Module 3 (`src/quantization/fake_quant.py`)**: Pre-Forward Hook dynamic fake-quantization injector (symmetric quantization with 99.99% clipping aligned with physical $W8A8Linear$; supports on-the-fly hot swapping).
4. **Module 4 (`src/eval/ppl.py`)**: Official standard 2048 sliding-window Perplexity (PPL) evaluation engine with crash-resilient CSV logging (`log_result_to_csv`).
5. **Module 5 (`scripts/run_pipeline.py`)**: Matrix orchestration controller (iterates across `[Models] × [Datasets] × [N] × [Bits]` with dry-run mode, multi-node collision prevention, and proactive memory reclamation).

---

## 📊 2. Verified Benchmark Baselines & Milestones

### 1. Unquantized Official Golden Baseline (WikiText-2 PPL)
Evaluated strictly with 2048 context length and 512 stride:
- 👑 **Qwen 3.5 4B**: Full 578-step PPL = **`8.2480`** *(Project best)*
- 🥇 **Llama 3.2 3B**: Full 562-step PPL = **`9.5177`** *(Sub-10.0 barrier)*
- 🥈 **Qwen 3.5 2B**: Full 578-step PPL = **`10.6631`** *(Outperforms previous-gen 2.5 3B at 10.72)*
- 🥉 **Llama 3.2 1B**: Full 562-step PPL = **`11.4027`** *(Matches Meta official whitepaper)*
- 🌟 **Gemma 2 2B**: Full 564-step PPL = **`11.8346`** *(Consistent convergence)*

### 2. Quantization Empirical Milestones
- 🏆 **Llama 3.2 1B (W8A16)**: PPL = **`11.4300`** (Minimal degradation of **`+0.0273` / `+0.24%`**; memory halved at -49.2%).
- **Current Objective**: Executing matrix evaluations across multi-domain calibration sets (WikiText-2, C4, GSM8K, HumanEval) to measure W16A8 and W16A4 PPL behaviors.

---

## 🛠️ 3. Google Colab Execution SOP

Follow these standard operational procedures when running in Google Colab:

### Step 1: Synchronize Latest Repository Code
```bash
%cd /content/model_quantization
!git pull origin main
```

### Step 2: Mount Google Drive (Crash-Resilient Artifact Storage)
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/model_quantization
```

### Step 3: Choose Execution Mode

#### 🚀 Mode A (Recommended): Full Matrix Orchestrator
> Automatically collects multi-domain activations, injects fake-quantization pre-hooks, computes standard PPL, logs scores to `/content/drive/MyDrive/results_Node_A.csv`, and releases accelerator memory after every iteration.

```bash
# 1. Fast Dry-Run (1 model, 16 samples, 5 PPL steps; verifies pipeline in ~30 seconds)
!python scripts/run_pipeline.py

# 2. Formal Matrix Evaluation (4 datasets × 3 sample sizes × 2 bit configurations = 24 evaluations)
# Default max_steps=40 (Recommended: ~35-40 min on a single T4 GPU):
!python scripts/run_pipeline.py --full

# 250-step deep evaluation (~4.5 hours on single T4; recommended for multi-node split):
!python scripts/run_pipeline.py --full --max_steps 250

# 3. Dual-Node Acceleration: Split by Dataset Domain
# Node A: General language corpora (WikiText-2 + C4)
!python scripts/run_pipeline.py --full --node Node_A --datasets wikitext2 c4

# Node B: Mathematical reasoning & code (GSM8K + HumanEval)
!python scripts/run_pipeline.py --full --node Node_B --datasets gsm8k humaneval

# 4. Dual-Node Acceleration: Split by Model
!python scripts/run_pipeline.py --full --node Node_A --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct
!python scripts/run_pipeline.py --full --node Node_B --models Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B
```

---

### 📚 Dataset Roles: Calibration vs. Evaluation

| Dataset | Domain Focus | Calibration Role | Evaluation Role |
| :--- | :--- | :--- | :--- |
| **WikiText-2** | High-quality encyclopedic text | Standard general-text activation distribution ($N \in \{16, 128, 512\}$) | **Official Academic Golden Benchmark** (2048-window PPL) |
| **C4** | Diverse web-scraped crawl | Broad web-scale outlier channel profiling | Cross-domain generalization test |
| **GSM8K** | Grade-school math reasoning | Symbolic token clusters & reasoning spikes | Mathematical reasoning sensitivity analysis |
| **HumanEval** | Python algorithmic code | Rigid code syntax & indentation token distributions | Code generation quantization degradation |

> 💡 **Core Diagnostic Principle**:  
> By calibrating on distinct domains (`wikitext2`, `c4`, `gsm8k`, `humaneval`) to harvest domain-specific scales, and testing uniformly against the international standard (WikiText-2 Test PPL), we quantify the impact of domain semantic drift on quantization robustness.

---

#### 📦 Mode B: Batch Download Models to Storage
```bash
!python scripts/download_model.py --all --save_dir /content/drive/MyDrive/models
```

#### 🔬 Mode C: Standalone Offline Calibration (Harvesting .pt Artifacts)
```bash
# 1. WikiText-2 calibration
!python scripts/run_calibration.py --model_id Qwen/Qwen3.5-2B --dataset wikitext2 --output_dir /content/drive/MyDrive/model_quantization/calibration_results

# 2. C4 web crawl calibration
!python scripts/run_calibration.py --model_id Qwen/Qwen3.5-2B --dataset c4 --output_dir /content/drive/MyDrive/model_quantization/calibration_results

# 3. GSM8K math reasoning calibration
!python scripts/run_calibration.py --model_id Qwen/Qwen3.5-2B --dataset gsm8k --output_dir /content/drive/MyDrive/model_quantization/calibration_results

# 4. HumanEval code calibration
!python scripts/run_calibration.py --model_id Qwen/Qwen3.5-2B --dataset humaneval --output_dir /content/drive/MyDrive/model_quantization/calibration_results
```

#### 📊 Mode D: Standalone Single Model PPL Evaluation
```bash
!python -m src.eval.ppl --model_id meta-llama/Llama-3.2-1B-Instruct --max_steps 40
```

---

## ⚠️ 4. Production Environment Safety Rules

1. **Maintain Architectural Integrity**:
   Core logic under `src/` and `configs/` has been verified with unit tests. Do not overwrite or redefine classes inside notebook cells.
2. **Proactive VRAM Reclamation**:
   During long-context (2048 tokens) evaluation, avoid storing models or large activation tensors in global variables. Always invoke `del model`, `torch.cuda.empty_cache()`, and `gc.collect()`.
3. **Crash-Resilient Logging**:
   All evaluation metrics must append to CSV files (`/content/drive/MyDrive/results_{node_name}.csv`) to guard against session timeouts.

---

## 🗺️ 5. Research Roadmap & Verification Milestones

### 📌 Phase 1 (Delivered Baseline): Multi-Domain × Multi-Model Benchmark Matrix
- **Core Objective**: Quantify empirical quantization baselines across open-source edge LLMs (Llama 3.2 1B/3B, Qwen 3.5 2B/4B, Gemma 2 2B) and calibration domains.
- **Evaluation Spectrum**: FP16/BF16 Baseline, physical W8A16, emulated W16A8, emulated W16A4.
- **Diagnostic Takeaway**: Map activation outlier topologies and identify the mathematical breakdown threshold under uniform A4 quantization.

### 🔬 Phase 2 (Proposed Verification Matrix): Advanced Orthogonal Rotations
- **Core Objective**: Verify whether orthogonal rotation transforms (SpinQuant / QuaRot) can suppress deep outlier spikes and recover extreme low-bit accuracy.
- **Architecture**:
  1. **Orthogonal Transform**: Walsh-Hadamard Transform (WHT) and learnable Cayley rotation matrices $R \in \mathbb{R}^{d \times d}$ ($R^T R = I$).
  2. **Feature Space Rotation**: Apply rotations prior to Attention ($Q, K, V, O$) and MLP (Gate, Up, Down) linear projections, diffusing outlier channel energy uniformly across dimensions.
  3. **Extreme Low-Bit PTQ**: Evaluate W4A4, W4A8, and W3A3 configurations.
  4. **Ablation Benchmark**: Quantify PPL recovery of `Standard PTQ` vs `QuaRot (Hadamard)` vs `SpinQuant (Cayley Rotation)`.
