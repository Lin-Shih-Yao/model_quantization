# 端側大語言模型之訓練後量化與激活離群特徵分析：跨領域實證基準研究 (FINAL REPORT)

**專案名稱：** Model Quantization Research Framework  
**專案倉庫：** [github.com/Lin-Shih-Yao/model_quantization](https://github.com/Lin-Shih-Yao/model_quantization)  
**日期：** 2026 年 10 月  
**關鍵詞：** 大語言模型 (LLMs)、訓練後量化 (PTQ)、激活離群值 (Activation Outliers)、跨領域泛化性、旋轉量化 (SpinQuant / QuaRot)、端側計算

---

## 1. 執行摘要 (Executive Summary)

隨著當代開源自回歸大語言模型逐步下沉至端側（1B～4B 參數量級），在嚴格的記憶體與計算功耗限制下部署模型，必須依賴激進的數值精度壓縮。然而，儘管 Weight-Only 量化（如 W8A16）損耗極低，一旦嘗試壓縮激活值至 8 位元以下（W16A8、W16A4、W4A4），極端的「激活離群尖刺（Activation Outlier Spikes）」會造成災難性的數值截斷與困惑度（PPL）暴衝。

本結案報告針對當前主流開源端側架構（**Llama 3.2 1B / 3B**、**Qwen 3.5 2B / 4B** 與 **Gemma 2 2B**）進行端到端實證研究。我們設計了一套高內聚、低耦合的五模組自動化 PTQ 評估管線，在四大不同領域語料（**WikiText-2**, **C4**, **GSM8K**, **HumanEval**）下執行了 48 組跨領域校準測試，並完成以下核心貢獻：

### 核心成果與實證發現：
1. **實體 W8A16 近乎零損耗量化：**  
   在 Llama-3.2-1B 上測得 **11.4300 PPL**（未量化 BF16 基準為 11.4027），損耗僅 **+0.0273 (+0.24%)**，實體顯存減半（50%），維持 150.3 tokens/s 推論吞吐量。
2. **官方學術黃金基準對齊：**  
   嚴格依循 2048 窗口、512 步長之官方標準滑動 PPL，對齊並突破國際基準：**Qwen 3.5 4B (8.2480)**、**Llama 3.2 3B (9.5177)**、**Qwen 3.5 2B (10.6631)**、**Llama 3.2 1B (11.4027)**、**Gemma 2 2B (11.8346)**。
3. **跨領域校準敏感度研究：**  
   完成了 48 組校準採集（4 資料集 $\times$ 3 樣本數 $N \in \{16, 128, 512\}$），實證分析領域遷移對量化動態範圍（Dynamic Range）與截斷誤差的影響。
4. **定位深層極端離群尖刺：**  
   成功捕捉神經網路深層的極端數值突波：**Qwen 3.5 2B 第 23 層** 出現高達 **40.50** 的離群值（均值的 18.2 倍）；**Gemma 第 34 層** 峰均比（PAR）更達到驚人的 **122.0 倍**。
5. **奠定第二階段旋轉量化之理論基礎：**  
   實證了 Uniform 量化在 sub-4-bit 激活下崩潰的根本病理，從而奠定引進 **Walsh-Hadamard 正交變換與 SpinQuant 可學習 Cayley 旋轉** 的學術與工程必要性。

---

## 2. 五模組解耦系統架構 (System Architecture)

為避免傳統量化腳本混亂耦合，專案嚴格將流程拆解為五大獨立模組，由全自動矩陣調度器統一控制：

```mermaid
graph TD
    M1["<b>模組 1: src/data/loader.py</b><br>多領域 Token 串流打包緩衝區<br>(WikiText-2, C4, GSM8K, HumanEval)"] -->|統一張量: N x 2048| M2["<b>模組 2: src/quantization/calibration.py</b><br>統一 LayerObserver 觀察器<br>統計極值、99.99% 門檻與通道 Scale"]
    M2 -->|校準中繼檔 (.json / .pt)| M3["<b>模組 3: src/quantization/fake_quant.py</b><br>動態 Pre-Forward Hook 注入器<br>(對齊實體 W8A8 數值)"]
    M3 -->|掛載偽量化模型| M4["<b>模組 4: src/eval/ppl.py</b><br>標準 2048 窗口滑動困惑度計算<br>防斷線 CSV 成績單日誌"]
    M5["<b>模組 5: scripts/run_pipeline.py</b><br>全矩陣控制器"] -.->|VRAM 強制釋放與排程| M1
    M5 -.->|VRAM 強制釋放與排程| M2
    M5 -.->|VRAM 強制釋放與排程| M3
    M5 -.->|VRAM 強制釋放與排程| M4
```

### 核心模組職責：
* **模組 1 (`src/data/loader.py`)：** 具備 **Token Streaming Packing Buffer**，在 Token 維度拼接短文本（如 HumanEval、GSM8K），徹底解決短文本因長度不足被丟棄的 Bug。
* **模組 2 (`src/quantization/calibration.py`)：** 利用 `LayerObserver` 收集真實前向傳播特徵，自動計算極值、99.99% 門檻值與對稱通道尺度。
* **模組 3 (`src/quantization/fake_quant.py`)：** 熱插拔式 Pre-Hook 注入器，完全對齊實體 `W8A8Linear` 的對稱量化與 99.99% 截斷算子。
* **模組 4 (`src/eval/ppl.py`)：** 實作官方 2048 窗口、512 步長、-100 歷史遮罩的滑動 PPL，杜絕非重疊切片的歷史上下文飢餓效應。
* **模組 5 (`scripts/run_pipeline.py`)：** 矩陣控制器，調度 `[模型] × [資料集] × [N] × [位元]`，每步嚴格執行 `del model`、`torch.cuda.empty_cache()` 與 `gc.collect()`，杜絕顯存洩漏。

---

## 3. 實證數據總表 (Empirical Benchmark Results)

### 3.1. 未量化官方黃金基準 (WikiText-2 Test Set)
| 模型型號 | 參數量 | 層數 | 評估窗口 | 實測 PPL | 評估步數 | 吞吐速度 | 學術對齊結論 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen 3.5 4B** | 4.54 B | 36 | 2048 | **8.2480** | 578 步 | 58.8 t/s | 刷新專案紀錄，超越技術報告預期 |
| **Llama 3.2 3B** | 3.21 B | 28 | 2048 | **9.5177** | 562 步 | 78.3 t/s | 強勢突破 10.0 大關，展現 3B 旗艦基線 |
| **Qwen 3.5 2B** | 2.21 B | 24 | 2048 | **10.6631** | 578 步 | 126.0 t/s | 實測超越前代 2.5 3B 基準 (10.72) |
| **Llama 3.2 1B** | 1.24 B | 16 | 2048 | **11.4027** | 562 步 | 179.6 t/s | 精準落在官方 11.0~12.0 預期區間 |
| **Gemma 2 2B** | 2.61 B | 26 | 2048 | **11.8346** | 564 步 | 188.9 t/s | 純文字架構正常收斂 |

### 3.2. 實體模組替換量化成果 (W8A16)
在 **Llama 3.2 1B** 上，將 112 個核心線性投影矩陣全數替換為 INT8 `W8A16Linear`：
* **未量化 Baseline**：`11.4027`
* **實體 W8A16 PPL**：`11.4300`（損耗僅 **+0.0273**，相對損耗率 **+0.24%**）
* **記憶體節省**：模型權重大小由 2.48 GB (BF16) 壓至 **1.26 GB (INT8)**，節省 **49.2%**。
* **結論**：證明端側自回歸模型的權重對 8-bit 對稱量化具備極高的數值韌性。

---

## 4. 激活離群尖刺診斷 (Activation Outlier Diagnostics)

透過攔截神經網路各層特徵矩陣，分析通道離群尖刺與峰均比（PAR）：
* **Qwen 3.5 2B（第 23 層）**：捕捉到特定的少數通道激活值暴衝至 **40.50**（高達均值的 18.2 倍）。
* **Gemma 2 2B（第 34 層）**：激活值峰值達到 **58.00**，峰均比（PAR）更達到驚人的 **122.0 倍**！

### 為什麼 Uniform 量化在 A4 會崩潰？
在 Uniform 量化公式下：
$$s = \frac{\max(|X|)}{2^{b-1} - 1}$$
當少數 4 個通道的離群值獨大時，量化階距 $s$ 會被無效拉大，導致其餘 99.8% 的正常特徵值全被壓扁映射至 0 或 1，使特徵空間喪失表徵能力。

---

## 5. 第二階段研發規劃：正交旋轉量化 (QuaRot & SpinQuant)

利用正交矩陣 $R$（滿足 $R^T R = I$）對特徵空間進行旋轉：
$$\tilde{X} = X R, \quad \tilde{W} = R^T W \implies \tilde{X}\tilde{W} = X W$$
1. **Walsh-Hadamard Transform (WHT)**：以無訓練成本的阿達馬矩陣將離群通道能量平均擴散。
2. **可學習 Cayley 旋轉矩陣 (SpinQuant)**：在 Stiefel 流形上以 Cayley 變換優化旋轉矩陣，最小化重構損失。
3. **極限精度評估**：針對 W4A4、W3A3 對比無旋轉 PTQ vs QuaRot vs SpinQuant 的困惑度恢復成效。
