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
    專職載入 Tokenizer 並確保 pad_token 存在。
    - 優先嘗試純文字 AutoTokenizer。
    - 若模型架構僅提供 Processor（如部分多模態變體），則降級自 Processor 擷取 Tokenizer。
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
    專職安全載入 Causal LM 權重並部署至指定硬體 (CUDA / MPS / CPU)。
    - CUDA：預設直接映射至當前 GPU，相容自訂量化層替換；亦支援傳入 device_map='auto'。
    - MPS / CPU：採用手動 .to(device) 避免 Accelerate 顯存衝突與 Kernel Panic。
    """
    resolved_path = resolve_model_path(model_id_or_path)
    actual_device = device or get_optimal_device()
    actual_dtype = dtype or torch_dtype or torch.bfloat16

    # 若未自訂 device_map，單卡 CUDA 建議直接指派 "cuda" 避免 hooks 干擾量化替換
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


load_model = load_causal_model


def load_pure_text_model_and_tokenizer(
    model_id_or_path: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """
    統一對外協調入口：循序載入裝置、Tokenizer 與 Causal LM 模型。
    """
    target_device = device or get_optimal_device()
    tokenizer = load_tokenizer(model_id_or_path)
    model = load_causal_model(model_id_or_path, device=target_device, dtype=dtype, **kwargs)
    return model, tokenizer


# 向後相容別名
load_model_and_tokenizer = load_pure_text_model_and_tokenizer
