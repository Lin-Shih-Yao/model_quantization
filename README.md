# Model Quantization Analysis & Advanced Rotation Framework

這是一個專注於**大語言模型（LLM）量化分析**與**進階正交旋轉量化技術（Rotation-based Quantization）**的研究與評估框架。

本專案旨在系統性評估現代開源主流模型在不同精度與量化策略下的表現，並深入探討利用旋轉變換（如 Walsh-Hadamard Transform、SpinQuant 等）消除激活值異常值（Activation Outliers）對極限低位元量化（如 W4A4、W3A3）的改善效果。

---

## 🎯 專案階段目標

### 第一階段：多模型 $\times$ 多精度量化基準分析 (Baseline & Precision Benchmark)
- **橫向比較**：在相同精度（如 W8A16, W4A16, W8A8, W4A4, FP8）下，分析不同模型架構對量化誤差的抗性。
- **縱向分析**：繪製各模型在 FP16 $\rightarrow$ FP8 $\rightarrow$ INT8 $\rightarrow$ INT4 $\rightarrow$ INT3 $\rightarrow$ INT2 的困惑度（PPL）與精度衰減曲線。
- **硬體效能**：評估不同量化策略對顯存佔用（VRAM/RAM）、推理延遲（Latency）與吞吐量的影響。

### 第二階段：進階旋轉量化技術 (Advanced Rotation Techniques)
- **異常值抑制**：利用正交旋轉矩陣 $R$（滿足 $R^T R = I$）對 Transformer 的 Attention 與 MLP 激活特徵空間進行旋轉變換，將極端異常值能量均勻分散。
- **極限精度評估**：針對 W4A4、W4A8、W3A3 等挑戰性極高的權重與激活量化，對比標準 PTQ vs 旋轉量化（QuaRot / SpinQuant）的恢復效果與消融分析。

---

## 🔬 目標分析模型 (Target Models)

專案選取三種具備代表性架構的主流開源端側模型進行深度對比分析：

### 1. Qwen 系列 (Alibaba Cloud)
* [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
* [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
* **架構特點**：SwiGLU 激活函數、Grouped Query Attention (GQA)、RoPE 位置編碼。

### 2. Llama 系列 (Meta)
* [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
* [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
* **架構特點**：128k 大詞表、標準 RMSNorm、GQA、學術界最通用的標準 Baseline 架構。

### 3. Gemma 系列 (Google)
* [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
* [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
* **架構特點**：Google 端側架構（GeGLU 激活、Logit Soft-Capping、滑動窗口注意力），適合與 LLaMA 架構進行異常值敏感度對比。

---

## 📁 專案目錄結構

```
model_quantization/
├── README.md                  # 專案說明文件
├── requirements.txt           # 核心依賴清單
├── .gitignore                 # Git 忽略配置
├── src/
│   ├── __init__.py
│   ├── quantization/          # 量化核心模組 (Uniform Quantizer, QuantLinear, FP8)
│   │   └── __init__.py
│   ├── models/                # 模型載入與架構替換 (Model Patcher)
│   │   └── __init__.py
│   └── eval/                  # 評估 Pipeline (PPL 計算, Profiler)
│       └── __init__.py
├── scripts/                   # 執行與實驗 CLI 腳本
│   └── __init__.py
└── tests/                     # 單元測試 (驗證量化數值誤差與正交性)
    └── __init__.py
```

---

## 🚀 快速開始 (Quickstart)

### 1. 環境建置
本專案已針對 Apple Silicon (Mac M 系列晶片 MPS 加速) 與 CUDA GPU 進行適配：

```bash
# 建立並啟用虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝核心依賴
pip install -r requirements.txt
```

### 2. 核心依賴一覽
- **深度學習與模型**：`torch`, `transformers`, `accelerate`, `datasets`
- **科學計算與旋轉變換**：`scipy`, `numpy`
- **分析與視覺化**：`matplotlib`, `tqdm`, `pytest`, `pyyaml`
