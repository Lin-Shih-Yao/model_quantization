"""
Task Registry for Downstream Generation Benchmarks.
"""

from . import gsm8k, humaneval

TASK_REGISTRY = {
    "gsm8k": gsm8k,
    "humaneval": humaneval,
}


def get_task_module(task_name: str):
    """Returns the task module implementing load_samples() and score_records()."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unsupported task: {task_name} (Available: {list(TASK_REGISTRY.keys())})")
    return TASK_REGISTRY[task_name]

