"""
Module 5: Main Pipeline Controller
Orchestrates the entire quantization benchmark matrix:
[Models] x [Datasets] x [N_samples] x [Bits]
Features:
- Dry Run test_mode mechanism
- Dual-Node collision-safe logging
- Strict VRAM survival cleanup after every bit and every model
"""

import argparse
import gc
import os
import sys
import torch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import CONFIG
from src.data import get_calibration_dataloader
from src.quantization import (
    collect_calibration_stats,
    save_calibration_stats_json,
    FakeQuantizer,
)
from src.eval.ppl import evaluate_ppl, log_result_to_csv
from src.models.loader import load_pure_text_model_and_tokenizer, resolve_model_path


def main(
    test_mode: bool = False,
    node_override: str | None = None,
    models_override: list[str] | None = None,
    datasets_override: list[str] | None = None,
    max_steps_override: int | None = None,
):
    """
    Main controller function: executes matrix benchmark experiments, enforces VRAM management,
    and logs crash-resilient results to CSV.
    """
    # --------------------------------------------------------------------------
    # 1. Test Mode Interception (Dry Run Mechanism)
    # --------------------------------------------------------------------------
    if test_mode:
        print("\n" + "=" * 70)
        print("🧪 Running in Test Mode (Fast Dry-Run)...")
        print("  • Model Count  : 1 model")
        print("  • Dataset      : wikitext2")
        print("  • Samples      : N=16")
        print("  • Quant Bits   : [8]")
        print("  • PPL Steps    : max_steps=5")
        print("=" * 70 + "\n")

        default_model = models_override[0] if models_override else (CONFIG["models"][0] if CONFIG["models"] else "meta-llama/Llama-3.2-1B-Instruct")
        models_to_run = [default_model]
        datasets_to_run = datasets_override or ["wikitext2"]
        n_samples_to_run = [16]
        bits_to_run = [8]
        max_steps = max_steps_override or 5
    else:
        models_to_run = models_override or CONFIG["models"]
        if not models_to_run:
            raise ValueError("❌ CONFIG['models'] is empty! Please configure at least one model ID.")
        datasets_to_run = datasets_override or CONFIG["datasets"]
        n_samples_to_run = CONFIG["N_samples"]
        bits_to_run = CONFIG["bits"]
        max_steps = max_steps_override or CONFIG.get("max_steps", 40)

    # --------------------------------------------------------------------------
    # 2. Global & Environment Configuration
    # --------------------------------------------------------------------------
    node_name = node_override or CONFIG.get("node_name", "Node_A")
    seq_len = CONFIG.get("seq_len", 2048)

    # Resolve CSV output path (supports Google Drive and local paths)
    if os.path.exists("/content/drive/MyDrive"):
        csv_log_path = f"/content/drive/MyDrive/results_{node_name}.csv"
    else:
        csv_log_path = f"./results_{node_name}.csv"

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"🖥️ Execution Node: {node_name} | Device: {device.upper()} | Precision: {dtype}")
    print(f"📊 Results will be persisted to: {csv_log_path}\n")

    # --------------------------------------------------------------------------
    # 3. Core Matrix Execution Loops (Models -> Datasets -> N -> Bits)
    # --------------------------------------------------------------------------
    for model_id in models_to_run:
        print(f"\n{'='*70}")
        print(f"🚀 [Loading Target Model] {model_id} ...")
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
                # (a) Harvest Layer Extrema: Module 1 -> Module 2 -> Save JSON
                # --------------------------------------------------------------
                print(f"\n🔍 [Harvesting Extrema] Model: {model_id} | Dataset: {dataset_name} | N={n} ...")
                calib_tensor = get_calibration_dataloader(
                    tokenizer=tokenizer,
                    dataset_name=dataset_name,
                    n_samples=n,
                    seq_len=seq_len,
                )

                layer_stats = collect_calibration_stats(model, calib_tensor, device=device)
                del calib_tensor

                # Save JSON stats (filename includes node_name for multi-node safety)
                save_calibration_stats_json(
                    stats=layer_stats,
                    model_name=model_id,
                    dataset_name=dataset_name,
                    n_samples=n,
                    node_name=node_name,
                )

                for bit in bits_to_run:
                    bit_str = f"W16A{bit}"
                    print(f"\n🧪 [Fake-Quant Evaluation] Injecting {bit_str} (N={n}, {dataset_name}) | max_steps={max_steps} ...")

                    # ----------------------------------------------------------
                    # (b) Emulation & Evaluation: Inject Pre-Hook -> Compute PPL -> Log CSV
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
                    # (c) VRAM Survival Cleanup: Detach Hook & Purge Accelerator Cache
                    # ----------------------------------------------------------
                    fake_quantizer.detach()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()

        # Cleanly delete model and tokenizer when model matrix finishes
        print(f"\n🧹 [Model Lifecycle Teardown] Reclaiming memory for {model_id}...")
        del model, tokenizer, fake_quantizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        print("✨ VRAM reset cleanly. Ready for next model.\n")

    print("\n" + "=" * 70)
    print("🎉 [Pipeline Complete] All matrix benchmarks evaluated and persisted!")
    print(f"📊 Detailed scorecard: {csv_log_path}")
    print("=" * 70 + "\n")


def parse_args():
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(description="LLM Quantization Benchmark Matrix Pipeline")
    parser.add_argument("--full", action="store_true", help="Execute full benchmark matrix (defaults to dry-run test mode)")
    parser.add_argument("--test_mode", action="store_true", help="Force fast test mode (N=16, steps=5)")
    parser.add_argument("--node", type=str, default=None, help="Override node name (e.g. Node_A, Node_B)")
    parser.add_argument("--models", nargs="+", default=None, help="Override target models list")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["wikitext2", "c4", "humaneval", "gsm8k"],
        help="Override calibration datasets list (e.g. wikitext2 c4)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override sliding PPL steps (e.g. 40 steps ~ 40 min; 250 steps ~ 4.5 hrs)",
    )
    return parser.parse_args()


# ==============================================================================
# Execution Entrypoint
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    is_test = False if args.full else True
    main(
        test_mode=is_test,
        node_override=args.node,
        models_override=args.models,
        datasets_override=args.datasets,
        max_steps_override=args.max_steps,
    )
