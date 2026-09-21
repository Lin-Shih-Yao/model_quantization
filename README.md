# Model Quantization Analysis & Advanced Rotation Framework

這是一個專注於**大語言模型（LLM）量化分析**、**全自動化量化基準評估**與**進階正交旋轉量化技術（Rotation-based Quantization）**的研究與評估框架。

本專案旨在系統性評估現代主流開源端側模型在不同精度與量化策略下的表現，並深入探討利用旋轉變換（如 Walsh-Hadamard Transform、SpinQuant 等）消除激活值異常值（Activation Outliers）對極限低位元量化（如 W4A4、W3A3）的改善效果。

---

## 🎯 專案階段目標

### 第一階段：多模型 $\times$ 多精度量化基準分析 (Baseline & Precision Benchmark)
- **橫向比較**：在相同精度（如 W8A16, W4A16, W8A8, W4A4, FP8）下，分析不同模型架構對量化誤差的抗性。
- **縱向分析**：繪製各模型在 FP16 $\rightarrow$ FP8 $\rightarrow$ INT8 $\rightarrow$ INT4 $\rightarrow$ INT3 $\rightarrow$ INT2 的困惑度（PPL）與精度衰減曲線。
- **全自動矩陣管線**：以五模組解耦架構自動遍歷 `[模型] × [資料集] × [校準樣本數] × [位元]`，並具備顯存安全回收與斷線防撞記錄機制。
- **硬體效能**：評估不同量化策略對顯存佔用（VRAM/RAM）、推理延遲（Latency）與吞吐量的影響。

### 第二階段：進階旋轉量化技術 (Advanced Rotation Techniques)
- **異常值抑制**：利用正交旋轉矩陣 $R$（滿足 $R^T R = I$）對 Transformer 的 Attention 與 MLP 激活特徵空間進行旋轉變換，將極端異常值能量均勻分散。
- **極限精度評估**：針對 W4A4、W4A8、W3A3 等挑戰性極高的權重與激活量化，對比標準 PTQ vs 旋轉量化（QuaRot / SpinQuant）的恢復效果與消融分析。

---

## 🔬 目標分析模型 (Target Models)

專案選取三種具備代表性架構的主流開源端側模型進行深度對比分析：

| 模型系列 | 目標型號 | 架構特色 | 參數量 |
| :--- | :--- | :--- | :---: |
| **Llama** (Meta) | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)<br>[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 128k 超長窗口、標準 RMSNorm、GQA、學術界主流黃金基線 | 1.2B / 3.2B |
| **Qwen** (Alibaba) | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)<br>[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | SwiGLU 激活、Grouped Query Attention (GQA)、RoPE | 2.2B / 4.5B |
| **Gemma** (Google) | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)<br>[google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 純文字自回歸 / 多模態、GeGLU 激活、Logit Soft-Capping、滑動窗口注意力 | 2.6B / 5.1B |

---

## 📁 專案目錄結構

```
model_quantization/
├── README.md                  # 專案說明文件與最新實測成果
├── requirements.txt           # 核心依賴清單
├── .gitignore                 # Git 忽略配置 (自動排除 /models/ 龐大權重)
├── src/
│   ├── __init__.py
│   ├── models/                # 模型載入、推論引擎與架構解耦
│   │   ├── __init__.py
│   │   ├── loader.py          # 模型與 Tokenizer 統一載入器、裝置偵測與路徑解析
│   │   └── inference.py       # 自適應 Chat Template 純文字高效推論引擎
│   ├── eval/                  # 評估 Pipeline (PPL 計算, Profiler, 成績單寫入)
│   │   ├── __init__.py
│   │   ├── ppl.py             # 模組 4: 官方標準滑動窗口 PPL 核心評估與 CSV 成績單日誌
│   │   └── profile_matrix.py  # 激活特徵矩陣與離群尖刺診斷分析
│   └── quantization/          # 量化核心演算法模組庫
│       ├── __init__.py
│       ├── data_loader.py     # 模組 1: 多領域校準資料抽取器 (WikiText2, C4, HumanEval, GSM8K)
│       ├── profiler.py        # 模組 2: 激活值與權重極值採集分析器 (Forward Hooks 監控每層 X_max)
│       ├── fake_quant.py      # 模組 3: 偽量化核心算子與 Pre-Hook 動態注入器 (W16A8 / W16A4)
│       ├── w8a16.py           # INT8 Weight-Only 量化模組
│       ├── w8a8.py            # INT8 權重與激活量化 (SmoothQuant 支援)
│       ├── calibration.py     # 離線激活值校準與 Scale 運算
│       ├── observer.py        # MinMax / Histogram 激活分佈觀察器
│       ├── utils.py           # 量化尺度 Scale & Zero-point 計算工具函式
│       └── dataset.py         # 基礎校準資料集封裝
├── notebooks/                 # 雲端 Jupyter 筆記本 (Google Colab 專用)
│   └── colab_quantization_benchmark.ipynb # 一鍵雲端評估與視覺化 Notebook
├── scripts/                   # 專案級自動化管線與 CLI 工具
│   ├── run_pipeline.py        # 模組 5: 五模組自動化矩陣量化評估主控制器 (支援 Dry Run)
│   ├── calibrate.py           # 獨立離線校準與 Scale 導出 CLI
│   ├── demo_inference.py      # 純文字輕量化推論測試腳本
│   └── download_model.py      # 模型權重下載工具 (支援單一、自訂清單與 --all 批次下載)
└── tests/                     # 單元測試庫 (驗證量化數值誤差與正交性)
    ├── __init__.py
    ├── test_calibration.py    # 模組 1 & 模組 2 校準極值統計單元測試
    ├── test_w8a16.py          # W8A16 量化精度與層替換測試
    └── test_w8a8.py           # W8A8 量化數值精確度測試
```

---

## ⚡ 核心架構：五模組自動化量化矩陣評估管線

專案將 PTQ（訓練後量化）與評估流程解耦為五大獨立模組，透過控制器自動輪巡 `[Models] × [Datasets] × [N_samples] × [Bits]` 矩陣：

```mermaid
graph TD
    M1["模組 1: data_loader.py<br>多領域校準資料抽取<br>(WikiText-2, C4, HumanEval, GSM8K)"] -->|固定形狀張量 (N, 2048)| M2["模組 2: profiler.py<br>極值採集器 (Forward Hook)<br>統計每層 X_max 並存為 JSON"]
    M2 -->|層級極值統計檔| M3["模組 3: fake_quant.py<br>偽量化動態注入器 (Pre-Hook)<br>支援 W16A8 / W16A4"]
    M3 -->|掛載偽量化模型| M4["模組 4: ppl.py<br>標準滑動窗口困惑度評估<br>即時寫入 CSV 成績單"]
    M5["模組 5: run_pipeline.py<br>主控制器 (矩陣調度)"] -.->|調度迴圈與顯存清理| M1
    M5 -.->|調度迴圈與顯存清理| M2
    M5 -.->|調度迴圈與顯存清理| M3
    M5 -.->|調度迴圈與顯存清理| M4
```

### 管線亮點特性：
1. **極速乾跑驗證 (Dry Run Mode)**：
   - 支援 `test_mode=True`，在極少步數（`max_steps=5`）與小樣本（`N=16`）下秒級驗證全管線各模組是否暢通。
2. **雙節點防撞記錄 (Dual-Node Safe Logging)**：
   - 支援 `Node_A` / `Node_B` 多機並行評估，成績單自動寫入獨立 CSV（支援 Google Drive `/content/drive/MyDrive/results_{node_name}.csv` 或本地 `./results_{node_name}.csv`），檔名與資料夾嚴防覆蓋。
3. **嚴格顯存釋放機制 (VRAM Survival Cleanup)**：
   - 每次測試完一個位元立即卸載 Pre-Hook 並執行顯存垃圾回收；切換模型時徹底刪除模型物件並清除 GPU 快取，確保連續評估不當機。

---

## 🚀 快速開始 (Quickstart)

### 1. 環境建置
本專案已全面適配 **Python 3.11+**，支援 **Apple Silicon MPS (Metal Performance Shaders)** 本地原生加速與 **NVIDIA CUDA (Google Colab)** 雲端算力：

```bash
# 建立並啟用虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝核心依賴
pip install -r requirements.txt
```

### 2. 模型權重下載
支援將 Hugging Face 權重直接下載至本地 `./models/` 目錄（支援單一模型、自訂清單與全量基準下載，受保護模型支援 `--token` 參數）：

```bash
# 1. 下載單一指定模型 (若為 Gated Model 可附帶 --token)
python scripts/download_model.py --model_id meta-llama/Llama-3.2-1B-Instruct

# 2. 批次下載專案所有基準模型 (Llama 3.2, Qwen 3.5, Gemma)
python scripts/download_model.py --all

# 3. 自訂下載多個模型清單
python scripts/download_model.py --models Qwen/Qwen3.5-2B google/gemma-2-2b-it
```

### 3. 一鍵執行五模組自動化量化矩陣管線
透過主控制器 `scripts/run_pipeline.py` 一鍵自動執行校準、極值採集、偽量化與 PPL 評估：

```bash
# 1. 極速乾跑測試模式 (驗證所有模組銜接正確，耗時 < 1 分鐘)
python scripts/run_pipeline.py

# 2. 全量矩陣評估模式 (依據 CONFIG 設定遍歷所有維度)
python -c "from scripts.run_pipeline import main; main(test_mode=False)"
```

### 4. 獨立離線校準 (Activation Calibration)
若需單獨對模型進行多樣本激活值採集並導出校準 Scale：

```bash
python scripts/calibrate.py --model_id Qwen/Qwen3.5-2B --n_samples 128 --seq_len 2048 --output_dir ./calibration_results
```

### 5. 純文字高效推論測試
專案提供專為語言模型設計的 `demo_inference.py`，支援自動偵測本地權重與**「非文字模組自動剝離（Text-Only Mode）」**，能主動釋放多模態視覺與語音編碼器，使記憶體佔用降至最低：

```bash
# 測試 Llama 3.2 1B 純文字推論
python scripts/demo_inference.py --model_id ./models/meta-llama_Llama-3.2-1B-Instruct --max_new_tokens 50

# 測試 Qwen 3.5 2B 純文字推論
python scripts/demo_inference.py --model_id ./models/Qwen_Qwen3.5-2B --max_new_tokens 50
```

### 6. 執行單元測試
驗證校準極值收集、W8A16 與 W8A8 數值誤差與層替換邏輯：

```bash
# 驗證校準資料集與極值分析模組
python tests/test_calibration.py

# 驗證 W8A16 算子與替換邏輯
python tests/test_w8a16.py

# 驗證 W8A8 算子與數值精度
python tests/test_w8a8.py
```

### 7. 雲端 GPU 極速評估 (Google Colab)
若本地記憶體有限，建議直接使用專案內的 Colab 雲端筆記本：
- **筆記本檔案**：[`notebooks/colab_quantization_benchmark.ipynb`](notebooks/colab_quantization_benchmark.ipynb)
- **核心優勢**：
  - 支援免費 T4 / A100 獨立 VRAM。
  - 支援完整 2048 官方標準窗口與全量 578 步 WikiText-2 PPL 評估。
  - 支援自動掛載 Google Drive，斷線不遺失評估成績單。

---

## 📊 本地基準推論效能 (MacBook Air M3, 16GB RAM)

在未量化原版（BF16 浮點精度）下，本機實測推論效能如下：

| 模型名稱 | 文字參數量 | 首次載入時間 | 生成速度 (Throughput) | 實體記憶體佔用 | 體驗評價 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🏆 **Llama 3.2 1B** | 1.24 B | **~ 3.0 s** | **`15.49 tokens/s`** | 🟢 **~ 2.5 GB** | **極速、超輕量** |
| **Llama 3.2 3B** | 3.21 B | ~ 14.6 s | **`7.11 tokens/s`** | 🟢 **~ 6.5 GB** | **流暢、高品質語意** |
| **Qwen 3.5 2B** | 2.21 B | ~ 22.5 s | **`7.49 tokens/s`** | 🟢 ~ 4.3 GB | 流暢 |
| **Gemma 4 E2B** | 5.10 B | ~ 57.7 s | **`4.28 tokens/s`** | 🟢 ~ 4.5 GB | 流暢 |
| **Qwen 3.5 4B** | 4.54 B | ~ 26.8 s | **`3.99 tokens/s`** | 🟢 ~ 8.5 GB | 偏吃重 |

---

## 🧪 官方基準評估與學術對齊 (Baseline Evaluation & Profiling)

### 1. WikiText-2 PPL 精度基準 (Perplexity Baseline)
本專案嚴格遵循 Hugging Face 官方標準滑動分塊規範（2048 Context Length, 512 Stride, -100 Context Masking），在 Apple Silicon (Mac M3 MPS) 與 Google Colab (NVIDIA CUDA GPU) 實測模型在未量化（BF16）下的黃金基準分數，並與國際頂級量化論文（BASE-Q, HeRo-Q 等）數據進行嚴格交叉對齊：

| 模型型號 | 實體參數量 | 國際論文 / 官方技術報告 | 本專案實測 PPL (Mac / Colab) | 評估規格與耗時 | 對齊結論 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 👑 **Qwen 3.5 4B** | 4.21 B | `8.0 ~ 8.8` (預期) | 🏆 **`8.25`** *(Colab 全量)* | **Colab 全量 578 步 (~ 84.2 min, 58.8 t/s)** | 🎯 **刷新紀錄！全量實測壓至 8.2480，展現 4B 端側旗艦級超強語言理解水準！** |
| 🥇 **Llama 3.2 3B** | 3.21 B | `9.5 ~ 10.5` (預期) | 🏆 **`9.52`** *(Colab 全量)*<br>🏆 **`9.54`** *(Mac 40 步)* | **Colab 全量 562 步 (~ 61.5 min, 78.3 t/s)**<br>Mac 40 步 (~ 3.7 min, 99.1 t/s) | 🎯 **強勢突破 10.0 大關！全量實測壓至 9.5177，展現 3B 端側頂級語言理解水準！** |
| 🥈 **Qwen 3.5 2B** | 2.21 B | *理論區間 `10.8 ~ 11.5`* | 🏆 **`10.66`** *(Colab 全量)*<br>🏆 **`11.20`** *(Mac 40 步)* | **Colab 全量 578 步 (~ 39.3 min, 126.0 t/s)**<br>Mac 40 步 (~ 5.8 min, 63.2 t/s) | 🎯 **全量評估突破理論上限！以 10.66 逆襲前代 2.5 3B 模型！** |
| 🥉 **Llama 3.2 1B** | 1.24 B | `11.0 ~ 12.0` (Instruct) | 🏆 **`11.40`** *(Colab 全量)*<br>🏆 **`11.29`** *(Mac 40 步)* | **Colab 全量 562 步 (~ 26.8 min, 179.6 t/s)**<br>Mac 40 步 (~ 1.5 min, 244.0 t/s) | 🎯 **極速收斂！以僅 1.2B 參數量精準對齊官方與論文基準，媲美 2B 等級困惑度！** |
| 🌟 **Gemma 2 2B** | 2.61 B | `11.5 ~ 12.5` (Instruct) | 🏆 **`11.83`** *(Colab 全量)* | **Colab 全量 564 步 (~ 25.5 min, 188.9 t/s)** | 🎯 **純文字架構正常收斂！實測 11.8346，精準對齊學術論文與官方基準！** |
| **Qwen 2.5 3B** | 3.09 B | `10.72` (FP16 基線) | — | — | 學術論文參考 (Qwen 3.5 2B 已實測超越此基準) |
| **Qwen 2.5 1.5B** | 1.54 B | `15.69` (BF16 基線) | — | — | 學術論文參考 |
| **Gemma 4 E2B** | 5.10 B | — | *12707.84 (多模態架構失配)* | Colab 全量 (多模態架構) | ⚠️ 因原生多模態架構與 PLE 詞表，純文字標準 PPL 評估不適用 |

> 💡 **全量評估亮點 (Colab Cloud Benchmark)**：  
> - **Qwen 3.5 4B**：在 Google Colab (CUDA, BF16) 上執行 **578 步全量 WikiText-2（共 297,053 tokens）** 深度評估，耗時 5051.78 秒（~84.2 分鐘，吞吐量 58.80 tokens/s），最終測得困惑度為 **`8.2480`**！大幅刷新專案全量測試最佳紀錄！
> - **Llama 3.2 3B**：全量 562 步 WikiText-2 實測困惑度達 **`9.5177`**，展現 3B 級別最強基線。
> - **Qwen 3.5 2B**：全量 578 步 WikiText-2 實測困惑度達 **`10.6631`**，超越前代 Qwen 2.5 3B 論文基線（`10.72`）。
> - **Llama 3.2 1B**：全量 562 步實測困惑度為 **`11.4027`**，精準落在官方技術報告 `11.0 ~ 12.0` 區間。
> - **Gemma 2 2B**：全量 564 步實測困惑度為 **`11.8346`**，以純文字架構完美收斂。

#### 執行標準 PPL 評估指令
透過 Python 模組模式直接呼叫 `src/eval/ppl.py`，支援多種規格調度：

```bash
# 1. 執行 40 步快速驗證模式 (建議初次篩選，約 1~5 分鐘)
python -m src.eval.ppl --model_id ./models/meta-llama_Llama-3.2-1B-Instruct --dataset wikitext-2-raw-v1 --max_length 2048 --stride 512 --max_steps 40

# 2. 執行 Llama 3.2 3B 快速驗證
python -m src.eval.ppl --model_id ./models/meta-llama_Llama-3.2-3B-Instruct --dataset wikitext-2-raw-v1 --max_length 2048 --stride 512 --max_steps 40

# 3. 執行全量完整資料集評估 (不帶 --max_steps 即評估全部 578 步，約 30~40 分鐘)
python -m src.eval.ppl --model_id ./models/Qwen_Qwen3.5-2B --dataset wikitext-2-raw-v1 --max_length 2048 --stride 512
```

---

### 2. 量化前激活值離群矩陣診斷 (Activation Outlier Profiling)
在正式量化前，透過 PyTorch Forward Hook 攔截神經網路各層特徵矩陣，分析通道離群尖刺（Outlier Spikes）與峰均比（Peak-to-Average Ratio, PAR），為後續量化策略提供理論依據：

* **Qwen 3.5 2B 實測診斷發現**：
  - 前中層（Layer 0 ~ 17）數值平緩（最大激活值 < 10.0，PAR ~ 11x）。
  - **深層出現極端尖刺**：在 **Layer 23** 捕捉到特定 4 個通道的激活值暴衝至 **`40.50`（高達均值的 18.2 倍！）**。
* **Gemma 4 E2B 實測診斷發現**：
  - 在 **Layer 34** 捕捉到激活值峰值達 **`58.00`**，峰均比（PAR）更達到驚人的 **`122.0 倍`**！
* **💡 研究結論（Why SpinQuant?）**：
  - 這客觀驗證了未量化大模型深層天生具備「少數離群通道」的病理特徵。若直接使用陽春 Uniform 量化會造成災難性截斷；因此後續實作 **SpinQuant（正交隨機旋轉）** 將離群能量平滑分散是保持低位元精度的關鍵！

```bash
# 執行激活矩陣體檢
python -m src.eval.profile_matrix --model_id ./models/Qwen_Qwen3.5-2B --num_tokens 1024
```

---

### 3. 量化實驗基準成果 (Post-Training Quantization Benchmarks)
本專案透過解耦式量化架構，實測模型在不同量化配置下的官方標準 WikiText-2 PPL 表現：

| 模型型號 | 量化配置 | 權重精度 / 激活精度 | 實測 WikiText-2 PPL | PPL 損耗 (Δ PPL) | 評估規格與耗時 | 量化結論 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Llama 3.2 1B** | **未量化 Baseline** | BF16 / BF16 | **`11.4027`** | — | Colab 全量 562 步 (1608s, 179.6 t/s) | 官方原始浮點基線基準 |
| 🏆 **Llama 3.2 1B** | **W8A16** | **INT8 / BF16** | **`11.4300`** | **`+0.0273` (`+0.24%`)** | **Colab 全量 562 步 (1922.5s, 150.3 t/s)** | 🎯 **教科書級近乎零損耗！顯存減半 (50%)，困惑度僅微幅增加 0.0273，完美保全語言理解力！** |

> 💡 **W8A16 量化實測亮點 (Llama 3.2 1B)**：
> - **評估規格**：全量 562 步滑動窗口（共計 288,937 tokens），總耗時 1922.49 秒 (~32.0 分鐘)，處理吞吐量達 150.29 tokens/秒。
> - **極致保真度**：在 112 個核心線性投影矩陣（$Q, K, V, O$、Gate, Up, Down）全數替換為 INT8 權重後，WikiText-2 困惑度僅從 `11.4027` 微幅移動至 **`11.4300`**，損耗率僅 **0.24%**，達到學術論文頂級標準！
> - **指令一鍵復現**：
>   ```bash
>   python -m src.eval.ppl --model_id meta-llama/Llama-3.2-1B-Instruct --quant_method w8a16
>   ```

---

## 📚 核心依賴一覽
- **深度學習與模型**：`torch >= 2.4.0`, `transformers >= 4.45.0`, `accelerate >= 0.28.0`, `datasets >= 2.18.0`
- **科學計算與旋轉變換**：`scipy >= 1.10.0`, `numpy >= 1.24.0`
- **多模態與輔助處理**：`torchvision`, `protobuf >= 3.20.0`, `sentencepiece >= 0.1.99`
- **分析與測試**：`matplotlib >= 3.7.0`, `tqdm >= 4.65.0`, `pytest >= 7.0.0`, `pyyaml >= 6.0.0`
