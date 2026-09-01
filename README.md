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

| 模型系列 | 目標型號 | 架構特色 | 參數量 |
| :--- | :--- | :--- | :---: |
| **Qwen** (Alibaba) | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)<br>[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | SwiGLU 激活、Grouped Query Attention (GQA)、RoPE | 2.2B / 4.5B |
| **Gemma** (Google) | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)<br>[google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | GeGLU 激活、Logit Soft-Capping、滑動窗口注意力 | 5.1B / 7.5B |
| **Llama** (Meta) | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)<br>[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 128k 大詞表、標準 RMSNorm、GQA、學術界主流 Baseline | 1.2B / 3.2B |

---

## 📁 專案目錄結構

```
model_quantization/
├── README.md                  # 專案說明文件
├── requirements.txt           # 核心依賴清單
├── .gitignore                 # Git 忽略配置 (自動排除 models/ 龐大權重)
├── src/
│   ├── __init__.py
│   ├── quantization/          # 量化核心模組 (Uniform Quantizer, QuantLinear, FP8)
│   │   └── __init__.py
│   ├── models/                # 模型載入與架構替換 (Model Patcher, Layer Slicer)
│   │   └── __init__.py
│   └── eval/                  # 評估 Pipeline (PPL 計算, Profiler, Latency)
│       └── __init__.py
├── scripts/                   # 執行與實驗 CLI 腳本
│   ├── demo_inference.py      # 純文字輕量化推論測試腳本
│   ├── download_model.py      # 單一模型權重下載工具
│   └── download_all_models.py  # 批次模型下載工具
└── tests/                     # 單元測試 (驗證量化數值誤差與正交性)
    └── __init__.py
```

---

## 🚀 快速開始 (Quickstart)

### 1. 環境建置
本專案已全面適配 **Python 3.11+** 與 **Apple Silicon MPS (Metal Performance Shaders)** 原生硬體加速：

```bash
# 建立並啟用虛擬環境
source .venv/bin/activate

# 安裝核心依賴
pip install -r requirements.txt
```

### 2. 模型權重下載
支援將 Hugging Face 權重直接下載至本地 `./models/` 目錄：

```bash
# 下載單一指定模型
python scripts/download_model.py --model_id Qwen/Qwen3.5-2B

# 或批次下載所有目標模型
python scripts/download_all_models.py
```

### 3. 純文字高效推論測試
專案提供專為語言模型量化設計的 `demo_inference.py`，支援自動偵測本地權重與**「非文字模組自動剝離（Text-Only Mode）」**，能主動釋放多模態視覺與語音編碼器，使記憶體佔用降至最低，避免設備發熱：

```bash
# 測試 Qwen 3.5 2B 純文字推論
python scripts/demo_inference.py --model_id ./models/Qwen_Qwen3.5-2B --max_new_tokens 50

# 測試 Gemma 4 E2B 純文字推論
python scripts/demo_inference.py --model_id ./models/google_gemma-4-E2B-it --max_new_tokens 50

# 測試 Qwen 3.5 4B 純文字推論
python scripts/demo_inference.py --model_id ./models/Qwen_Qwen3.5-4B --max_new_tokens 50
```

---

## 📊 本地基準推論效能 (MacBook Air M3, 16GB RAM)

在未量化原版（BF16 浮點精度）下，本機實測推論效能如下：

| 模型名稱 | 文字參數量 | 首次載入時間 | 生成速度 (Throughput) | 實體記憶體佔用 |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen 3.5 2B** | 2.21 B | ~ 22.5 s | **7.49 tokens/s** | 🟢 ~ 4.3 GB (流暢) |
| **Gemma 4 E2B** | 5.10 B | ~ 57.7 s | **4.28 tokens/s** | 🟢 ~ 4.5 GB (流暢) |
| **Qwen 3.5 4B** | 4.54 B | ~ 26.8 s | **3.99 tokens/s** | 🟢 ~ 8.5 GB (流暢) |

---

## 📚 核心依賴一覽
- **深度學習與模型**：`torch >= 2.13`, `transformers >= 5.x`, `accelerate`, `datasets`
- **科學計算與旋轉變換**：`scipy`, `numpy`
- **多模態與輔助處理**：`torchvision`, `protobuf`, `sentencepiece`
- **分析與測試**：`matplotlib`, `tqdm`, `pytest`, `pyyaml`
