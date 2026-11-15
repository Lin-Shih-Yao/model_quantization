"""
HumanEval Python Code Generation Task (Loading & Sandboxed pass@1 Scoring).
"""

import ast
import multiprocessing
import re
from datasets import load_dataset


def load_samples(limit: int | None = None, seed: int = 42) -> list[dict]:
    """Loads HumanEval test problems."""
    del seed  # HumanEval uses fixed sequential ordering
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        ds = load_dataset("openai_humaneval", split="test")

    max_n = min(limit, len(ds)) if limit else len(ds)
    return [
        {
            "id": ds[idx]["task_id"],
            "prompt": (
                "Complete the following Python function. "
                "Output only valid Python code without extra explanation.\n\n"
                f"{ds[idx]['prompt']}"
            ),
            "raw_prompt": ds[idx]["prompt"],
            "ground_truth": ds[idx]["test"],
            "entry_point": ds[idx]["entry_point"],
        }
        for idx in range(max_n)
    ]


def _extract_python_code(prediction: str, raw_prompt: str) -> str:
    """Extracts Python code block and ensures the function signature is present."""
    code_match = re.search(r"```(?:python)?\n(.*?)```", prediction, re.DOTALL)
    code = code_match.group(1) if code_match else prediction
    if raw_prompt and raw_prompt.strip().splitlines()[-1] not in code:
        code = f"{raw_prompt}\n{code}"
    return code


def _exec_worker(code: str, test_code: str, entry_point: str, queue: multiprocessing.Queue) -> None:
    """Executes HumanEval unit tests in an isolated child process."""
    try:
        namespace: dict = {}
        exec(code, namespace)
        exec(test_code, namespace)
        namespace["check"](namespace[entry_point])
        queue.put("pass")
    except SyntaxError:
        queue.put("syntax_error")
    except Exception:
        queue.put("fail")


def _evaluate_sample(code: str, test_code: str, entry_point: str, timeout_sec: float = 5.0) -> str:
    """Runs syntax check and unit tests with a hard timeout."""
    try:
        ast.parse(code)
    except SyntaxError:
        return "syntax_error"

    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_worker, args=(code, test_code, entry_point, queue))
    proc.start()
    proc.join(timeout=timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "timeout"
    return queue.get() if not queue.empty() else "fail"


def score_records(records: list[dict]) -> dict:
    """Computes pass@1 and syntax_error_rate over HumanEval records."""
    passed = 0
    syntax_errors = 0
    for r in records:
        code = _extract_python_code(r["prediction"], r.get("raw_prompt") or "")
        status = _evaluate_sample(code, r["ground_truth"], r["entry_point"])
        if status == "pass":
            passed += 1
        elif status == "syntax_error":
            syntax_errors += 1

    total = len(records)
    return {
        "primary_metric": "pass_at_1",
        "primary_value": round(passed / total, 4),
        "secondary_metrics": {"syntax_error_rate": round(syntax_errors / total, 4)},
    }

