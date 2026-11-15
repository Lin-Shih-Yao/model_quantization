def __getattr__(name):
    if name in ("evaluate_ppl", "log_result_to_csv"):
        from .ppl import evaluate_ppl, log_result_to_csv
        mapping = {"evaluate_ppl": evaluate_ppl, "log_result_to_csv": log_result_to_csv}
        return mapping[name]
    if name == "profile_activation_matrix":
        from .profile_matrix import profile_activation_matrix
        return profile_activation_matrix
    if name in ("run_evaluation_suite", "score_jsonl", "rescore_all_jsonl"):
        from .downstream import rescore_all_jsonl, run_evaluation_suite, score_jsonl
        mapping = {
            "run_evaluation_suite": run_evaluation_suite,
            "score_jsonl": score_jsonl,
            "rescore_all_jsonl": rescore_all_jsonl,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "evaluate_ppl",
    "log_result_to_csv",
    "profile_activation_matrix",
    "run_evaluation_suite",
    "score_jsonl",
    "rescore_all_jsonl",
]
