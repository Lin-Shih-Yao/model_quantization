# 專案開發歷程回顧與工程問題排查白皮書 (Development Journey & Problem Solving)

> **文件定位**：本文件記錄此端側大模型量化專案在開發歷程中遭遇的 **5 大核心工程挑戰、根本原因（Root Cause）、架構防禦策略** 以及對應之 **Git Commit 與原始碼位置**。專為接手維護、技術面試或技術匯報提供快速脈絡導讀。

---

## 🧭 開發脈絡總覽 (Timeline Overview)

本專案在實作過程中並非單純呼叫開源 API，而是經歷了從 **「單純跑通推論」 $\to$ 「遭遇硬體與數值崩潰」 $\to$ 「重構五模組解耦系統」 $\to$ 「發現深層神經網路激活尖刺」** 的完整工程迭代歷程：

```mermaid
flowchart LR
    A["階段 1: 基礎跑通<br>(推論/下載腳本)"] --> B["階段 2: 遭遇瓶頸<br>(OOM / Padding 污染 / 404)"]
    B --> C["階段 3: 解耦重構<br>(5 模組架構 / 顯存保險絲)"]
    C --> D["階段 4: 數值對齊<br>(FakeQuant vs 物理 W8A8)"]
    D --> E["階段 5: 病理診斷<br>(逐層 PAR / 尖刺定位)"]
```

---

## 🛠️ 五大核心工程問題與排查歷程

### 1. 記憶體爆炸（OOM）與長序列推論崩潰
* **遭遇症狀**：
  在 Google Colab (16GB T4 GPU) 或本機 Apple M3 (16GB UMA) 進行 2048 窗口長序列評估時，PyTorch 底層 `CachingAllocator` 產生大量不可回收之內部顯存碎片。模型在連續評估 30~50 步後，突然觸發未預期之 CUDA Out-of-Memory (OOM) 或系統記憶體恐慌閃退。
* **根因分析 (Root Cause)**：
  自回歸模型長文本前向傳播的 Activation 與 KV-Cache 暫存量大，單純依賴 Python 垃圾回收無法釋放顯存池中的未釋放 block。
* **架構防禦與實作**：
  - **逐步顯存主動回收機制**：在滑動窗口每前進一步（Step）時，主動調用顯式記憶體釋放：
    ```python
    del outputs, input_ids, target_ids
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()
    ```
  - **雙重保險絲 (Safety Circuit Breaker)**：在評估核心加入即時 RAM 監控，當系統記憶體水位超過 85% 時，主動中斷以保護宿主機不當機。
* **對應 Commit 與檔案**：
  - Commit: `549eff1`, `ecb33cf`
  - 核心程式碼：[src/eval/ppl.py](file:///Users/wanghuiyu/Desktop/model_quantization/src/eval/ppl.py#L102-L115)

---

### 2. 短文本資料集補零（Padding）污染自注意力機制
* **遭遇症狀**：
  當拿程式碼資料集（HumanEval，平均 $< 200$ tokens）或小學數學題（GSM8K）作為校準語料時，若採用傳統補零（Zero-padding）強制對齊至 2048 長度，收集到的激活值分佈會出現異常的「零值波谷」，導致算出的量化尺度（Scale）嚴重偏小，量化後的通用推理直接崩潰。
* **根因分析 (Root Cause)**：
  Transformer 的 Self-Attention 機制會受到 Padding Token 的人為干擾；零填充降低了語料的實際資訊熵，破壞了原本連續自然語言的數值動態分佈。
* **架構防禦與實作**：
  - **無損無填充串流打包 (Token Streaming Packing Buffer)**：放棄整題補零，改在 Token 維度實施連續滾動拼接。多篇文章與題目像串肉串般首尾相連，僅在累積滿 2048 tokens 時切出一個飽和張量 `(1, 2048)`，確保 100% 資訊真實度。
* **對應 Commit 與檔案**：
  - Commit: `27a5feb`, `451455f`
  - 核心程式碼：[src/data/loader.py](file:///Users/wanghuiyu/Desktop/model_quantization/src/data/loader.py#L90-L97)

---

### 3. 開源資料集上游依賴變動（HuggingFace 載入死鎖）
* **遭遇症狀**：
  評估基準核心依賴的 `wikitext-2-raw-v1` 在 HuggingFace Datasets 某次上游更新後，呼叫 `load_dataset("wikitext", ...)` 會發生 404 Not Found 或連線超時，導致自動化 CI/CD 與 Colab Pipeline 整個中斷。
* **根因分析 (Root Cause)**：
  依賴單一公開資料集名稱容易受到 HuggingFace 社群權限更動或倉庫重新命名的衝擊。
* **架構防禦與實作**：
  - **多重 Fallback 備援載入器**：在所有資料集載入點包裝 `try-except` 策略，優先嘗試官方最新鏡像，失敗時自動無縫降級至備援源（例如先試 `wikitext`，失敗自動切換 `Salesforce/wikitext`；GSM8K 先試 `openai/gsm8k`，失敗切換 `gsm8k`）。
* **對應 Commit 與檔案**：
  - Commit: `aaabeaf`
  - 核心程式碼：[src/data/loader.py](file:///Users/wanghuiyu/Desktop/model_quantization/src/data/loader.py#L53-L70)、[src/eval/ppl.py](file:///Users/wanghuiyu/Desktop/model_quantization/src/eval/ppl.py#L59)

---

### 4. 偽量化模擬（FakeQuant）與實體層替換（W8A8）數值對不齊
* **遭遇症狀**：
  早期透過 PyTorch Hook 做的動態偽量化（FakeQuantizer，用浮點模擬 INT8 精度損失）算出來的困惑度，與真正替換為實體類別（`W8A8Linear`）的模型表現存在顯著誤差（$\Delta \text{PPL} > 0.5$），無法作為可信的快速模擬工具。
* **根因分析 (Root Cause)**：
  兩套系統採用的「動態範圍截斷（Clipping）」與「零點偏移（Zero-point）」計算公式未統一。一端用了非對稱量化（Asymmetric），另一端用了對稱量化（Symmetric），且分位數截斷門檻（99.99%）未能完全共享。
* **架構防禦與實作**：
  - **收斂統一校準架構 (Unified LayerObserver)**：將極值、99.99% 分位數門檻值及尺度計算徹底收斂於 `src/quantization/observer.py`。
  - 將 FakeQuant 與 W8A8 全面統一為 **對稱量化（Symmetric Quantization）**，共用相同的 Scale 計算邏輯，使模擬結果能 100% 反映物理替換之數值行為。
* **對應 Commit 與檔案**：
  - Commit: `c65144c`, `41f6b28`
  - 核心程式碼：[src/quantization/observer.py](file:///Users/wanghuiyu/Desktop/model_quantization/src/quantization/observer.py)、[src/quantization/fake_quant.py](file:///Users/wanghuiyu/Desktop/model_quantization/src/quantization/fake_quant.py)

---

### 5. 極限低位元（4-bit 激活 A4）崩潰之深層數值病理
* **遭遇症狀**：
  將權重或激活值由 8-bit（W8A8）進一步壓低至 4-bit（W16A4）時，各大模型在標準測試集上的困惑度發生毀滅性塌陷（PPL 飆升至幾千，完全失去語言組織能力）。
* **根因分析與診斷洞察 (Diagnostic Breakthrough)**：
  - 傳統評估只看最後端對端的 PPL，無法知道神經網路內部在哪一層壞掉。
  - 專案特地開發了激活特徵矩陣診斷工具（`src/eval/profile_matrix.py`），逐層監控所有隱藏層通道的**極值與峰均比（PAR, Peak-to-Average Ratio）**。
  - **重大發現**：崩潰並非全網路均勻退化，而是**深層少數特定通道出現極端異常尖刺（Outlier Channel Spikes）**！
    - **Qwen 3.5 2B**：前 17 層最大值平緩（$< 10.0$），但在 **Layer 23** 突發高達 **`40.50`** 的異常高壓（達均值的 18.2 倍）。
    - **Gemma 2 2B**：在 **Layer 34** 激活值直衝 **`58.00`**，峰均比高達 **`123.4×`**！
  - **結論**：當只有 16 個量化階距（4-bit = $2^4$）時，均勻量化的 Step Size 被這幾個少數極端尖刺強行撐大，導致其餘 99.9% 的正常數值全部被擠壓進同一個階距，產生毀滅性的量化截斷誤差。
* **對應 Commit 與檔案**：
  - Commit: `06ed375`
  - 核心程式碼：[src/eval/profile_matrix.py](file:///Users/wanghuiyu/Desktop/model_quantization/src/eval/profile_matrix.py)

---

## 📚 專案工程沉澱與核心價值 (Takeaway for Interviews)

當被問及「這個專案有何工程深度？」時，可總結為以下 3 點：

1. **具備端側硬體思維**：不只調用現成套件，而是針對邊緣設備的記憶體頻寬限制（Memory-Bandwidth Bound）實作了顯存生命週期重置與串流打包防禦。
2. **具備數值根因分析能力 (Root-Cause Profiling)**：不僅量測到「4-bit 崩潰」，更進一步定位到「Qwen L23 與 Gemma L34 的 PAR 尖刺」，用數據解釋均勻階距擴張之病理。
3. **具備多維度評估閉環**：涵蓋語言流暢度（WikiText-2 PPL）、數學推理（GSM8K 雙指標）、程式編寫（HumanEval 沙盒 pass@1）與重複鬼打牆退化監控（Repetition Rate）。
