"""
Module 5: Main Pipeline Controller
Orchestrates the entire quantization benchmark matrix:
[Models] x [Datasets] x [N_samples] x [Bits]
Features:
- Dry Run test_mode mechanism
- Dual-Node collision-safe logging
- Strict VRAM survival cleanup after every bit and every model
"""

import gc
import os
import sys
import torch

# 確保專案根目錄在 sys.path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quantization.data_loader import CONFIG, get_calibration_dataloader
from src.quantization.profiler import CalibrationProfiler, save_stats_to_json
from src.quantization.fake_quant import FakeQuantizer
from src.eval.ppl import evaluate_ppl, log_result_to_csv
from src.models.loader import load_pure_text_model_and_tokenizer, resolve_model_path


def main(test_mode: bool = False):
    """
    主控制器函式：依序執行全流程矩陣實驗，並嚴格管控顯存與防斷線存檔。
    """
    # --------------------------------------------------------------------------
    # 1. 測試模式攔截 (Dry Run 機制)
    # --------------------------------------------------------------------------
    if test_mode:
        print("\n" + "=" * 70)
        print("🧪 目前為 Test Mode，僅進行極速乾跑測試...")
        print("  • 模型數量: 1 款")
        print("  • 測試資料: wikitext2")
        print("  • 樣本數量: N=16")
        print("  • 量化位元: [8]")
        print("  • PPL 步數: max_steps=5")
        print("=" * 70 + "\n")

        default_model = CONFIG["models"][0] if CONFIG["models"] else "meta-llama/Llama-3.2-1B-Instruct"
        models_to_run = [default_model]
        datasets_to_run = ["wikitext2"]
        n_samples_to_run = [16]
        bits_to_run = [8]
        max_steps = 5
    else:
        models_to_run = CONFIG["models"]
        if not models_to_run:
            raise ValueError("❌ CONFIG['models'] 為空！請填入至少一個待測模型名稱。")
        datasets_to_run = CONFIG["datasets"]
        n_samples_to_run = CONFIG["N_samples"]
        bits_to_run = CONFIG["bits"]
        max_steps = 250

    # --------------------------------------------------------------------------
    # 2. 全域與環境設定
    # --------------------------------------------------------------------------
    node_name = CONFIG.get("node_name", "Node_A")
    seq_len = CONFIG.get("seq_len", 2048)

    # 決定成績單 CSV 路徑 (支援 Google Drive 與本地目錄)
    if os.path.exists("/content/drive/MyDrive"):
        csv_log_path = f"/content/drive/MyDrive/results_{node_name}.csv"
    else:
        csv_log_path = f"./results_{node_name}.csv"

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"🖥️ 執行節點: {node_name} | 硬體設備: {device.upper()} | 精度: {dtype}")
    print(f"📊 成績單將記錄至: {csv_log_path}\n")

    # --------------------------------------------------------------------------
    # 3. 核心迴圈架構 (Models -> Datasets -> N -> Bits)
    # --------------------------------------------------------------------------
    for model_id in models_to_run:
        print(f"\n{'='*70}")
        print(f"🚀 [載入模型] {model_id} ...")
        print(f"{'='*70}")

        resolved_path = resolve_model_path(model_id)
        model, tokenizer = load_pure_text_model_and_tokenizer(
            resolved_path,
            device=device,
            dtype=dtype,
        )
        fake_quantizer = FakeQuantizer()

        for dataset_name in datasets_to_run:
            for n in n_samples_to_run:
                # --------------------------------------------------------------
                # (a) 收集極值：呼叫模組 1 取得資料 -> 呼叫模組 2 收集並存檔
                # --------------------------------------------------------------
                print(f"\n🔍 [極值收集] 模型: {model_id} | 資料集: {dataset_name} | N={n} ...")
                calib_tensor = get_calibration_dataloader(
                    tokenizer=tokenizer,
                    dataset_name=dataset_name,
                    n_samples=n,
                    seq_len=seq_len,
                )

                profiler = CalibrationProfiler()
                layer_stats = profiler.collect_stats(model, calib_tensor, device=device)
                del calib_tensor

                # 存檔為 JSON (檔名內含 node_name 防撞)
                save_stats_to_json(
                    stats=layer_stats,
                    model_name=model_id,
                    dataset_name=dataset_name,
                    n_samples=n,
                    node_name=node_name,
                )

                for bit in bits_to_run:
                    bit_str = f"W16A{bit}"
                    print(f"\n🧪 [偽量化評估] 注入 {bit_str} (N={n}, {dataset_name}) | max_steps={max_steps} ...")

                    # ----------------------------------------------------------
                    # (b) 偽量化與評估：注入 Pre-Hook -> 計算 PPL -> 寫入 CSV
                    # ----------------------------------------------------------
                    fake_quantizer.attach(model, layer_stats, bits=bit)

                    ppl_res = evaluate_ppl(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=seq_len,
                        stride=512,
                        max_steps=max_steps,
                    )
                    ppl_score = ppl_res["ppl"] if isinstance(ppl_res, dict) else ppl_res

                    log_result_to_csv(
                        log_path=csv_log_path,
                        node_name=node_name,
                        model_id=model_id,
                        calib_dataset=dataset_name,
                        n_samples=n,
                        bit_str=bit_str,
                        ppl_score=ppl_score,
                    )

                    # ----------------------------------------------------------
                    # (c) VRAM 保命清理：立刻卸載 Hook，清空顯存快取
                    # ----------------------------------------------------------
                    fake_quantizer.detach()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()

        # 該模型全數測完後，從記憶體徹底刪除 model 與 tokenizer
        print(f"\n🧹 [模型結束清理] 徹底釋放 {model_id} 之顯存與物件...")
        del model, tokenizer, fake_quantizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        print("✨ 顯存已徹底還原乾淨，準備載入下一個模型。\n")

    print("\n" + "=" * 70)
    print("🎉 [Pipeline 執行完畢] 所有矩陣實驗已成功歸檔！")
    print(f"📊 完整成績單請查看: {csv_log_path}")
    print("=" * 70 + "\n")


# ==============================================================================
# 執行區塊
# ==============================================================================
if __name__ == "__main__":
    # 預設啟動極速乾跑測試模式 (驗證流程 100% 暢通)
    main(test_mode=True)
