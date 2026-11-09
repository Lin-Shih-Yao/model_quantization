"""
Unified Model & Tokenizer Loader Module
Safely loads causal language models without peak VRAM surges and without breaking forward dependencies.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedTokenizerBase,
)


def get_optimal_device() -> str:
    """Returns 'cuda' if GPU available, 'mps' on Apple Silicon, otherwise 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_path(model_id: str, base_dir: str = "./models") -> str:
    """Resolves model ID to local directory if already downloaded, otherwise returns model ID."""
    if os.path.exists(model_id):
        return os.path.abspath(model_id)

    short_name = model_id.replace("/", "_")
    local_dir = os.path.abspath(os.path.join(base_dir, short_name))
    if os.path.exists(local_dir):
        return local_dir

    return model_id


def load_tokenizer(model_id_or_path: str, **kwargs) -> PreTrainedTokenizerBase:
    """
    Loads Tokenizer and ensures pad_token is defined.
    - Attempts standard text AutoTokenizer first.
    - If model is multimodal or packaged as a Processor, falls back to extracting tokenizer from AutoProcessor.
    """
    resolved_path = resolve_model_path(model_id_or_path)
    tok_kwargs = {"trust_remote_code": True, **kwargs}
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_path, **tok_kwargs)
    except (ValueError, KeyError, AttributeError):
        processor = AutoProcessor.from_pretrained(resolved_path, **tok_kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_causal_model(
    model_id_or_path: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    torch_dtype: torch.dtype | None = None,
    device_map: str | dict | None = None,
    **kwargs,
) -> torch.nn.Module:
    """
    Loads Causal LM weights safely onto targeted hardware (CUDA / MPS / CPU).
    - CUDA: Defaults to direct device placement to facilitate custom quantized layer swapping.
    - MPS / CPU: Uses explicit .to(device) to prevent Accelerate dispatch conflicts.
    """
    resolved_path = resolve_model_path(model_id_or_path)
    actual_device = device or get_optimal_device()
    actual_dtype = dtype or torch_dtype or torch.bfloat16

    # For single GPU CUDA, prefer explicit 'cuda' mapping unless custom device_map specified
    if device_map is None and actual_device == "cuda":
        resolved_device_map = "cuda"
    else:
        resolved_device_map = device_map

    load_kwargs = {
        "torch_dtype": actual_dtype,
        "low_cpu_mem_usage": True,
        "device_map": resolved_device_map,
        "trust_remote_code": True,
        **kwargs,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(resolved_path, **load_kwargs)
    except (ValueError, KeyError, AttributeError):
        model = AutoModelForImageTextToText.from_pretrained(resolved_path, **load_kwargs)

    if resolved_device_map is None and actual_device:
        model = model.to(actual_device)

    model.eval()
    return model


def load_pure_text_model_and_tokenizer(
    model_id_or_path: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """
    Unified public coordinator: sequentially loads tokenizer and causal LM model onto target hardware.
    """
    target_device = device or get_optimal_device()
    tokenizer = load_tokenizer(model_id_or_path)
    model = load_causal_model(model_id_or_path, device=target_device, dtype=dtype, **kwargs)
    return model, tokenizer
