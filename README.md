# LLM Post-Training Quantization & Activation Outlier Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Device: CUDA / Apple Silicon MPS](https://img.shields.io/badge/Hardware-CUDA%20%7C%20MPS-green.svg)]()

> **Note**: For the traditional Chinese version of this documentation, please see [README_zh.md](README_zh.md).  
> For the comprehensive technical findings and empirical analysis, refer to [FINAL_REPORT.md](FINAL_REPORT.md).

An end-to-end empirical research and evaluation framework for **Large Language Model (LLM) Post-Training Quantization (PTQ)**, systematic cross-domain calibration analysis, and **activation outlier diagnostics** across edge-deployable LLMs.

---

## 🎯 Key Objectives

1. **Precision & Architecture Sensitivity Matrix:**  
   Empirically assess quantization resilience across heterogeneous edge architectures (Llama 3.2, Qwen 3.5, Gemma 2) under multiple bit-width configurations (W8A16, W16A8, W16A4).
2. **Cross-Domain Calibration Generalization:**  
   Evaluate how diverse calibration distributions (**WikiText-2**, **C4**, **GSM8K**, **HumanEval**) impact cross-domain linguistic perplexity.
3. **Activation Outlier Localization:**  
   Trace and quantify activation channel spikes and peak-to-average ratios (PAR) across transformer layers.
4. **Diagnostic Evidence for Rotation Techniques:**  
   Quantify numerical degradation under naive uniform quantization to theoretically motivate and guide future rotation-based methods (e.g., QuaRot, SpinQuant).

---

## 🔬 Target Edge Models

We evaluate representative edge-deployable causal language models:

| Model Family | Model ID | Parameters | Architecture Highlights |
| :--- | :--- | :---: | :--- |
| **Llama** (Meta) | `meta-llama/Llama-3.2-1B-Instruct`<br>`meta-llama/Llama-3.2-3B-Instruct` | 1.2B / 3.2B | 128k context, standard RMSNorm, Grouped Query Attention (GQA), academic gold baseline |
| **Qwen** (Alibaba) | `Qwen/Qwen3.5-2B`<br>`Qwen/Qwen3.5-4B` | 2.2B / 4.5B | SwiGLU activation, RoPE embeddings, GQA |
| **Gemma** (Google) | `google/gemma-2-2b-it` | 2.6B | GeGLU activation, Logit soft-capping, sliding-window attention |

---

## ⚡ 5-Stage Decoupled Quantization Architecture

The repository enforces a clean, modular pipeline separating calibration extraction, simulated fake quantization, evaluation, and hardware telemetry:

```mermaid
graph TD
    M1["<b>Module 1: src/data/loader.py</b><br>Token Streaming Packing Buffer<br>(WikiText-2, C4, GSM8K, HumanEval)"] -->|Uniform Tensor: N x 2048| M2["<b>Module 2: src/quantization/calibration.py</b><br>Unified LayerObserver<br>Captures extrema, 99.99% quantiles & channel scales"]
    M2 -->|Calibration Profiles (.json / .pt)| M3["<b>Module 3: src/quantization/fake_quant.py</b><br>Dynamic Pre-Forward Hook Injector<br>(Symmetric W16A8 / W16A4 emulation)"]
    M3 -->|Hooked Causal Model| M4["<b>Module 4: src/eval/ppl.py</b><br>Standard 2048-Window Sliding PPL<br>Crash-resilient CSV logging"]
    M5["<b>Module 5: scripts/run_pipeline.py</b><br>Matrix Orchestrator & Controller"] -.->|VRAM survival cleanup & dispatch| M1
    M5 -.->|VRAM survival cleanup & dispatch| M2
    M5 -.->|VRAM survival cleanup & dispatch| M3
    M5 -.->|VRAM survival cleanup & dispatch| M4
```

### Architectural Highlights:
* **Token Streaming Packing Buffer:** Seamlessly concatenates short-text entries (e.g., HumanEval, GSM8K) into continuous 2048-token chunks without artificial pad tokens.
* **Symmetric Clipping Operator:** Aligns dynamic fake quantization with physical `W8A8Linear` kernels using $99.99\%$ quantile thresholds.
* **Strict VRAM Survival Policy:** Executes explicit garbage collection and cache eviction (`del model`, `torch.cuda.empty_cache()`, `gc.collect()`) after every evaluation step, preventing memory leaks during long runs.

---

## 📊 Empirical Baseline & PTQ Results

Evaluated on the full official **WikiText-2 Test Set** (Context length: 2048, Stride: 512, Target Mask: -100):

| Model | Parameters | Precision | Measured PPL | Degradation ($\Delta$ PPL) | VRAM Footprint | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen 3.5 4B** | 4.54 B | BF16 Baseline | **`8.2480`** | — | ~8.5 GB | 58.8 tokens/s |
| **Llama 3.2 3B** | 3.21 B | BF16 Baseline | **`9.5177`** | — | ~6.5 GB | 78.3 tokens/s |
| **Qwen 3.5 2B** | 2.21 B | BF16 Baseline | **`10.6631`** | — | ~4.3 GB | 126.0 tokens/s |
| **Llama 3.2 1B** | 1.24 B | BF16 Baseline | **`11.4027`** | — | ~2.5 GB | 179.6 tokens/s |
| 🏆 **Llama 3.2 1B** | 1.24 B | **W8A16 PTQ** | **`11.4300`** | **`+0.0273` (`+0.24%`)** | **~1.3 GB (-50%)** | **150.3 tokens/s** |
| **Gemma 2 2B** | 2.61 B | BF16 Baseline | **`11.8346`** | — | ~4.5 GB | 188.9 tokens/s |

> 💡 **W8A16 Milestone:** Replacing 112 projection matrices with INT8 linear layers produces textbook-grade compression: **0.24% perplexity delta with 50% memory savings**.

---

## 🔍 Activation Outlier Profiling

Through forward-hook activation profiling, we mapped internal activation spikes and peak-to-average ratios (PAR):
* **Qwen 3.5 2B (Layer 23):** Isolated channels surge to **`40.50`** (18.2× the layer mean).
* **Gemma 2 2B (Layer 34):** Peak activations reach **`58.00`** with a **`122.0×` Peak-to-Average Ratio**.

These empirical findings illustrate why standard uniform quantization suffers catastrophic collapse under 4-bit activations (A4) due to step-size dilation, providing the diagnostic foundation for future Phase 2 research into orthogonal feature rotations.

---

## 🗺️ Roadmap: Planned Phase 2 Research

> ⚠️ **Phase 2 Status**:  
> The methodologies below represent planned research directions designed to mitigate the activation spikes identified in Phase 1. They are not yet implemented in the current codebase.

* [ ] **Walsh-Hadamard Transform (WHT):** Implement training-free block-diagonal Hadamard transforms to distribute channel peak energy across hidden dimensions.
* [ ] **Learnable Cayley Rotations (SpinQuant Architecture):** Optimize orthogonal matrices over the Stiefel manifold to minimize per-layer reconstruction error.
* [ ] **Sub-4-Bit Activation Benchmarking:** Benchmark extreme precision targets (W4A4, W3A3) comparing unrotated vs. rotated representations.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Lin-Shih-Yao/model_quantization.git
cd model_quantization

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Model Weights
```bash
# Download single model (supports --token for gated models)
python scripts/download_model.py --model_id meta-llama/Llama-3.2-1B-Instruct

# Download all benchmark models
python scripts/download_model.py --all
```

### 3. Run Automated 5-Stage Quantization Pipeline
```bash
# Rapid dry run verification (~30 seconds)
python scripts/run_pipeline.py

# Full matrix evaluation (4 datasets x 3 N_samples x 2 bit-widths)
python scripts/run_pipeline.py --full

# Parallel execution across dual nodes
python scripts/run_pipeline.py --full --node Node_A --datasets wikitext2 c4
python scripts/run_pipeline.py --full --node Node_B --datasets gsm8k humaneval
```

### 4. Run Offline Activation Calibration
```bash
python scripts/run_calibration.py --model_id Qwen/Qwen3.5-2B --dataset wikitext2 --n_samples 128 --seq_len 2048
```

### 5. Run Unit Tests
```bash
python tests/test_calibration.py
python tests/test_fake_quant.py
python tests/test_w8a16.py
python tests/test_w8a8.py
```

---

## 📁 Repository Structure

```
model_quantization/
├── FINAL_REPORT.md            # Comprehensive project technical report
├── README.md                  # Main English project documentation
├── README_zh.md               # Traditional Chinese documentation
├── requirements.txt           # Core project dependencies
├── configs/
│   └── pipeline_config.py     # Matrix evaluation configuration
├── src/
│   ├── data/
│   │   └── loader.py          # Module 1: Multi-domain streaming data loader
│   ├── quantization/
│   │   ├── calibration.py     # Module 2: Unified activation observer & scale calculator
│   │   ├── observer.py        # LayerObserver tracking extrema, clipping & channel peaks
│   │   ├── fake_quant.py      # Module 3: Dynamic Pre-Forward Hook fake quantizer
│   │   ├── w8a16.py           # Physical INT8 weight-only linear layer
│   │   └── w8a8.py            # Physical INT8 weight & activation layer
│   ├── eval/
│   │   ├── ppl.py             # Module 4: Standard sliding-window perplexity evaluator
│   │   └── profile_matrix.py  # Activation outlier and spike diagnostics
│   └── models/
│       ├── loader.py          # Device-adaptive causal model and tokenizer loader
│       └── inference.py       # Pure-text inference engine
├── scripts/
│   ├── run_pipeline.py        # Module 5: Automated matrix orchestration controller
│   ├── run_calibration.py    # Standalone calibration CLI
│   └── download_model.py      # Unified model weight downloader
└── tests/                     # Unit test suites (Calibration, Fake Quant, W8A16, W8A8)
```

---

## 📜 References

1. **SmoothQuant:** Accurate and Efficient Post-Training Quantization for Large Language Models (*Xiao et al., 2023*)
2. **QuaRot:** Outlier-Free 4-Bit LLM Inference via Orthogonal Rotations (*Ashkboos et al., 2024*)
3. **SpinQuant:** LLM Quantization with Learned Rotations (*Liu et al., 2024*)
4. **Hugging Face Transformers:** Perplexity of Fixed-Length Models Benchmark Standards.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

