"""
Unified Text Generation & Inference Engine
Provides reusable generation functions (Chat template, sampling, throughput) across all supported LLM architectures.
"""

import time
import torch


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str = "mps",
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    system_prompt: str | None = None,
) -> dict:
    """
    Unified text generation function.
    Automatically applies model's official chat_template if available.

    Returns:
        dict: {
            "response": str,
            "input_tokens": int,
            "output_tokens": int,
            "latency_sec": float,
            "tokens_per_sec": float,
        }
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # 1. 套用官方對話模板 (Chat Template) 或退回純文字
    if getattr(tokenizer, "chat_template", None) is not None:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt

    # 2. Tokenize 輸入並送往目標設備
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # 3. 前向自回歸生成
    start_time = time.time()
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id or 0,
    }
    if do_sample and temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    elapsed_time = time.time() - start_time

    # 4. 解碼生成的新 Token
    generated_tokens = outputs[0][input_len:]
    num_generated = len(generated_tokens)
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    throughput = num_generated / elapsed_time if elapsed_time > 0 else 0.0

    return {
        "response": response_text,
        "input_tokens": input_len,
        "output_tokens": num_generated,
        "latency_sec": elapsed_time,
        "tokens_per_sec": throughput,
    }
