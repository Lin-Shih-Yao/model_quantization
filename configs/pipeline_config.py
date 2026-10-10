"""
Global Experiment Configuration for Quantization Pipeline (Module 5).
Configures target models, datasets, sample sizes, bits, and execution nodes.
"""

CONFIG = {
    "node_name": "Node_A",
    "models": [
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
    "datasets": ["wikitext2", "c4", "humaneval", "gsm8k"],
    "N_samples": [16, 128, 512],
    "bits": [8, 4],
    "seq_len": 2048,
}
