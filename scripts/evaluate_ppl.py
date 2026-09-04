"""
CLI Shortcut to run Perplexity (PPL) Evaluation.
Usage:
    python scripts/evaluate_ppl.py --model_id ./models/Qwen_Qwen3.5-2B
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.ppl import main

if __name__ == "__main__":
    main()
