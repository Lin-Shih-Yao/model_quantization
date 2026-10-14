# 🧭 Colab 雲端評估指南與專案進度手冊 (Colab AI & Gemini Context Guide)

> **這份文件的用途**：  
> 本文件專為 **Google Colab 執行環境** 與 **Colab 側邊欄 Gemini AI 助理** 設計。  
> 當切換到 Google Colab 時，Gemini 可藉由閱讀本手冊，即刻掌握**專案架構、最新進度、已修復事項、標準作業流程（SOP）與執行指令**，無縫接軌本地開發進度。

---

## 🎯 一、 專案核心定位與架構速覽

本專案專注於 **LLM 大語言模型量化（PTQ）**、**極限低位元（W4A4、W3A3）** 以及 **進階正交旋轉量化技術（SpinQuant / QuaRot）** 的系統性基準評估。

### 核心五模組自動化量化管線架構
專案已將量化與評估解耦為五大獨立模組，由控制器全自動調度：
1. **[模組 1] `src/data/loader.py`**：多領域校準資料抽取（WikiText-2, C4, HumanEval, GSM8K，統一切割 2048 窗口，支援樣本 $N \in \{16, 128, 512\}$）。全域實驗配置位於 `configs/pipeline_config.py`。
2. **[模組 2] `src/quantization/calibration.py`（與相容適配器 `profiler.py`）**：統一校準引擎（基於 `LayerObserver`，精準統計極值、99.99% 門檻截斷、通道峰值與 Symmetric/Asymmetric Scale，支援 `.pt` 與 `.json` 輸出）。
3. **[模組 3] `src/quantization/fake_quant.py`**：Pre-Forward Hook 偽量化動態注入器（支援對稱量化與 99.99% Clipping，數值完全對齊真實 $W8A8Linear$；支援多格式 `.pt` / `.json` / 記憶體字典直讀，即時熱插拔）。
4. **[模組 4] `src/eval/ppl.py`**：官方標準 2048 窗口滑動困惑度（PPL）評估，並具備防斷線成效日誌（`log_result_to_csv`）。
5. **[模組 5] `scripts/run_pipeline.py`**：全矩陣調度主控制器（讀取 `configs/pipeline_config.py`，自動輪巡 `[Models] × [Datasets] × [N] × [Bits]`，具備 Dry Run 模式、雙節點防撞與顯存強制回收）。

---

## 📊 二、 目前已完成之里程碑與基準數據 (Current Progress)

### 1. 未量化官方黃金基準 (WikiText-2 PPL Baseline)
所有數據均嚴格遵循 2048 窗口、512 Stride 官方規範實測對齊：
- 👑 **Qwen 3.5 4B**：實測全量 578 步 PPL = **`8.2480`** *(刷新專案最佳紀錄)*
- 🥇 **Llama 3.2 3B**：實測全量 562 步 PPL = **`9.5177`** *(突破 10.0 大關)*
- 🥈 **Qwen 3.5 2B**：實測全量 578 步 PPL = **`10.6631`** *(超越前代 2.5 3B 基準 10.72)*
- 🥉 **Llama 3.2 1B**：實測全量 562 步 PPL = **`11.4027`** *(精準對齊官方報告)*
- 🌟 **Gemma 2 2B**：實測全量 564 步 PPL = **`11.8346`** *(純文字架構正常收斂)*

### 2. 量化基準實測進度
- 🏆 **Llama 3.2 1B (W8A16)**：PPL = **`11.4300`**（損耗僅 **`+0.0273` / `+0.24%`**，顯存減半近乎零損耗）。
- **當前重點任務**：在 Colab GPU 上執行五模組自動化矩陣評估管線，採集多模型在多領域校準資料（WikiText-2, C4, GSM8K, HumanEval）下的偽量化（W16A8、W16A4）PPL 表現。

### 3. 最近完成的 Code Review 重構重點（Colab 請特別注意！）
- **校準與量化策略對齊統一**：Fake Quant 偽量化算子全面對齊實體 W8A8 的對稱截斷（Symmetric with 99.99% Clipping）；校準器全面收斂至 `LayerObserver` 與 `collect_calibration_stats()`，廢除重複的 Hook 代碼，實現真實模組替換與模擬偽量化的無縫統一。
- **資料模組統一與策略映射**：校準資料集載入整合至 `src/data/loader.py`，採用 Strategy Pattern 字典映射查表，並將全域設定抽離至 `configs/pipeline_config.py`。
- **下載腳本整合**：所有模型下載統一由 [`scripts/download_model.py`](file:///Users/wanghuiyu/Desktop/model_quantization/scripts/download_model.py) 處理（支援單一模型、`--all` 全量下載、`--models` 自訂清單）。
- **離線校準腳本更名**：已重新命名為 [`scripts/run_calibration.py`](file:///Users/wanghuiyu/Desktop/model_quantization/scripts/run_calibration.py)（原 `calibrate.py` 已廢棄）。
- **移除薄包裝腳本**：`scripts/evaluate_ppl.py` 與 `scripts/profile_matrix.py` 已刪除，單獨評估請使用 Python 標準模組指令：
  - `python -m src.eval.ppl ...`
  - `python -m src.eval.profile_matrix ...`
- **推論安全修復**：`src/models/inference.py` 已修正 `pad_token_id` 安全 fallback，並修復 Base 模型遺失 `system_prompt` 的問題。

---

## 🛠️ 三、 Colab 執行 SOP 指南（給 Colab Gemini 的操作指南）

在 Google Colab 中執行時，請嚴格按照以下流程進行：

### 步驟 1：同步最新代碼
```bash
# 切換至專案根目錄並拉取 GitHub 最新程式碼
%cd /content/model_quantization
!git pull origin main
```

### 步驟 2：掛載 Google Drive（防斷線與成績單備份）
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/model_quantization
```

### 步驟 3：選擇執行模式

#### 🚀 模式 A（推薦）：執行五模組全自動量化矩陣管線
> **說明**：自動進行極值採集、動態注入偽量化 Pre-Hook、計算 PPL，並即時將結果寫入 Google Drive 中的 `/content/drive/MyDrive/results_Node_A.csv`，每輪結束自動釋放 VRAM。

```bash
# 1. 極速乾跑測試 (Dry Run: 1 款模型, 16 樣本, 5 步 PPL, 驗證流程暢通約 1 分鐘)
!python scripts/run_pipeline.py

# 2. 正式全量評估 (依據 CONFIG 設定跑完整矩陣)
!python -c "from scripts.run_pipeline import main; main(test_mode=False)"
```

#### 📦 模式 B：批次下載模型至 Google Drive 或本地
```bash
# 下載專案全套基準模型
!python scripts/download_model.py --all --save_dir /content/drive/MyDrive/models
```

#### 🔬 模式 C：單獨執行離線校準 (Activation Calibration)
```bash
!python scripts/run_calibration.py --model_id Qwen/Qwen3.5-2B --output_dir /content/drive/MyDrive/model_quantization/calibration_results
```

#### 📊 模式 D：單獨快速測試特定模型 PPL
```bash
!python -m src.eval.ppl --model_id meta-llama/Llama-3.2-1B-Instruct --max_steps 40
```

---

## ⚠️ 四、 Colab 雲端環境核心安全守則

1. **嚴禁破壞性修改現有模組**：
   `src/` 與 `configs/` 目錄下的核心邏輯（`loader.py`, `src/data/loader.py`, `profiler.py`, `fake_quant.py`, `ppl.py`, `pipeline_config.py`）已經過本地單元測試驗證，切勿在 Colab 內自行覆寫或新增重複定義。
2. **顯存存活原則（VRAM Survival）**：
   在長序列（2048 Context Length）評估時，嚴禁在全域變數長時間駐留多個 Model 或 Large Tensor。每次評估結束務必呼叫 `del model`、`torch.cuda.empty_cache()` 與 `gc.collect()`。
3. **成績單保護原則**：
   所有量化成績單一律記錄至 CSV（檔案路徑優先採用 `/content/drive/MyDrive/results_{node_name}.csv`），以防 Colab 因執行逾時（Timeout）斷線遺失數據。
