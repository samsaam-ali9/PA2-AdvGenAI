from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B"


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        try:
            return torch.device(device)
        except (TypeError, RuntimeError):
            pass
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def load_model_and_tokenizer(
    checkpoint: str,
    *,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected_device = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if torch_dtype is None:
        torch_dtype = torch.float16 if selected_device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config_exists = Path(checkpoint).is_dir() and (Path(checkpoint) / "adapter_config.json").exists()
    if adapter_config_exists:
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "This checkpoint is a PEFT adapter; install peft to load it for inference."
            ) from exc
        model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch_dtype)
    model.to(selected_device)
    model.eval()
    return model, tokenizer



def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive.")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size cannot be negative.")
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when sampling.")
    if do_sample and not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1] when sampling.")

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if not isinstance(encoded, dict):
        encoded = dict(encoded)
    target_device = _model_device(model)
    encoded = {key: value.to(target_device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    if "input_ids" not in encoded or encoded["input_ids"].ndim != 2:
        raise ValueError("Tokenizer must return input_ids with shape [batch, sequence].")
    if encoded["input_ids"].shape[0] != 1:
        raise ValueError("generate_response expects exactly one prompt at a time.")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": pad_token_id,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)
    prompt_length = encoded["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_alpaca_eval_rows() -> list[dict[str, Any]]:

    try:
        from datasets import load_dataset

        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        return list(dataset)
    except Exception as dataset_error:
        try:
            from huggingface_hub import hf_hub_download

            source_path = hf_hub_download(
                repo_id="tatsu-lab/alpaca_eval",
                filename="alpaca_eval.json",
                repo_type="dataset",
            )
            with open(source_path, "r", encoding="utf-8") as handle:
                source = json.load(handle)
            if isinstance(source, dict):
                source = source.get("eval", source.get("data"))
            if not isinstance(source, list):
                raise ValueError("Official AlpacaEval JSON did not contain a list of prompts.")
            return source
        except Exception as hub_error:
            raise RuntimeError(
                "Could not load official AlpacaEval prompts via datasets or Hub JSON. "
                f"datasets error: {dataset_error}; Hub error: {hub_error}"
            ) from hub_error


def create_alpaca_eval_subset(
    source_rows: Sequence[dict[str, Any]] | None = None,
    *,
    n: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    
    if n <= 0:
        raise ValueError("n must be positive.")
    if source_rows is None:
        source_rows = _load_alpaca_eval_rows()
    source_rows = list(source_rows)
    if len(source_rows) < n:
        raise ValueError(f"Need {n} prompts, but the AlpacaEval source has {len(source_rows)}.")

  
    selected_indices = sorted(random.Random(seed).sample(range(len(source_rows)), n))
    subset = []
    seen_ids: set[str] = set()
    for output_index, source_index in enumerate(selected_indices):
        row = source_rows[source_index]
        if not isinstance(row, dict):
            raise ValueError(f"AlpacaEval row {source_index} is not a JSON-like object.")
        instruction = row.get("instruction", row.get("prompt"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"AlpacaEval row {source_index} has no instruction text.")
        row_id = str(row.get("id", f"alpaca-eval-seed{seed}-{output_index:04d}"))
        if row_id in seen_ids:
            raise ValueError(f"AlpacaEval IDs are not unique: {row_id}")
        seen_ids.add(row_id)
        subset.append(
            {
                "id": row_id,
                "instruction": instruction.strip(),
                "dataset": "alpaca_eval",
                "source_index": source_index,
            }
        )
    return subset


def generate_dataset_outputs(
    model: Any,
    tokenizer: Any,
    evaluation_rows: Sequence[dict[str, Any]],
    *,
    generator: str,
    dataset_name: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> list[dict[str, Any]]:
    
    if not generator.strip() or not dataset_name.strip():
        raise ValueError("generator and dataset_name must be non-empty strings.")
    from data_utils import format_prompt

    results = []
    seen_ids: set[str] = set()
    for index, row in enumerate(evaluation_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Evaluation row {index} must be an object.")
        instruction = row.get("instruction", row.get("prompt"))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Evaluation row {index} has no instruction/prompt.")
        row_id = str(row.get("id", f"{dataset_name}-{index:04d}"))
        if row_id in seen_ids:
            raise ValueError(f"Evaluation IDs are not unique: {row_id}")
        seen_ids.add(row_id)
        prompt = format_prompt(instruction, tokenizer)
        record = {
            "id": row_id,
            "instruction": instruction.strip(),
            "output": generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ),
            "generator": generator,
            "dataset": dataset_name,
            "generation_settings": {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
            },
        }
        for key in ("category", "expected_behavior", "source_index", "aspect"):
            if key in row:
                record[key] = row[key]
        results.append(record)
    return results


def save_generations(generations: Sequence[dict[str, Any]], path: str | Path) -> str:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(list(generations), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)
    return str(path)
