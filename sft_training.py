from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _encode(tokenizer: Any, text: str) -> list[int]:
    tokens = tokenizer.encode(str(text), add_special_tokens=False)
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return [int(token) for token in tokens]


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    return int(pad_id) if pad_id is not None else 0


def build_sft_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    *,
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:

    if max_length < 1:
        raise ValueError("max_length must be at least 1.")
    if max_prompt_length is not None and max_prompt_length < 1:
        raise ValueError("max_prompt_length must be positive when provided.")
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same number of rows.")
    if len(prompts) == 0:
        raise ValueError("Cannot build an empty language-model batch.")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    pad_id = _pad_token_id(tokenizer)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = _encode(tokenizer, prompt)
        response_ids = _encode(tokenizer, response)
        if eos_id is not None:
            response_ids.append(int(eos_id))

        if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]


        prompt_budget = max(0, max_length - 1)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []


        if not prompt_ids and max_length > 1:
            prefix_id = bos_id if bos_id is not None else eos_id
            if prefix_id is not None:
                prompt_ids = [int(prefix_id)]

        response_budget = max_length - len(prompt_ids)
        response_part = response_ids[:response_budget]
        sequence = prompt_ids + response_part
        if not sequence:
            fallback_id = bos_id if bos_id is not None else eos_id
            if fallback_id is None:
                raise ValueError(
                    "A row encoded to no tokens; provide non-empty text or a tokenizer "
                    "with BOS/EOS tokens."
                )
            sequence = [int(fallback_id)]
            prompt_ids = sequence
            response_part = []

        labels = [-100] * len(prompt_ids) + response_part
        input_rows.append(sequence)
        label_rows.append(labels)

    width = max(map(len, input_rows))
    input_ids = torch.full((len(input_rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(input_rows), width), dtype=torch.long)
    labels = torch.full((len(input_rows), width), -100, dtype=torch.long)
    for row_index, (tokens, targets) in enumerate(zip(input_rows, label_rows)):
        row_length = len(tokens)
        input_ids[row_index, :row_length] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row_index, :row_length] = 1
        labels[row_index, :row_length] = torch.tensor(targets, dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def compute_sft_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    #  Return mean cross-entropy over response tokens after causal shifting
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    batch = {key: value.to(_model_device(model)) for key, value in batch.items()}
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    shifted_logits = logits[:, :-1, :].contiguous().float()
    shifted_labels = batch["labels"][:, 1:].contiguous()
    valid_count = shifted_labels.ne(-100).sum()
    if valid_count.item() == 0:
        raise ValueError(
            "No response tokens remain after causal shifting; increase max_length "
            "or provide a scoreable response and context token."
        )
    loss_sum = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum / valid_count


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _grad_scaler(enabled: bool):

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _formatted_rows(
    rows: Sequence[dict[str, Any]], tokenizer: Any
) -> tuple[list[str], list[str]]:
    from data_utils import format_prompt

    prompts: list[str] = []
    responses: list[str] = []
    for index, row in enumerate(rows):
        prompt = row.get("instruction", row.get("prompt"))
        response = row.get("response", row.get("output", row.get("completion")))
        if prompt is None or response is None:
            raise ValueError(
                f"SFT row {index} needs instruction/prompt and response/output/completion."
            )
        prompts.append(format_prompt(str(prompt), tokenizer))
        responses.append(str(response))
    return prompts, responses


def evaluate_sft(
    model: torch.nn.Module,
    tokenizer: Any,
    validation_data: Sequence[dict[str, Any]],
    max_length: int = 512,
    batch_size: int = 2,
    use_amp: bool = True,
) -> float:
    if not validation_data:
        raise ValueError("validation_data must contain at least one example.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    was_training = model.training
    model.eval()
    device = _model_device(model)
    total_loss = 0.0
    total_tokens = 0
    prompts, responses = _formatted_rows(validation_data, tokenizer)
    try:
        for start in range(0, len(prompts), batch_size):
            batch = build_sft_batch(
                tokenizer,
                prompts[start : start + batch_size],
                responses[start : start + batch_size],
                max_length,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad(), _autocast(device, use_amp):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                shifted_logits = logits[:, :-1, :].contiguous().float()
                shifted_labels = batch["labels"][:, 1:].contiguous()
                token_count = int(shifted_labels.ne(-100).sum().item())
                if token_count:
                    loss_sum = F.cross_entropy(
                        shifted_logits.view(-1, shifted_logits.size(-1)),
                        shifted_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_loss += float(loss_sum.item())
                    total_tokens += token_count
    finally:
        if was_training:
            model.train()

    if total_tokens == 0:
        raise ValueError("Validation set contains no scoreable response tokens.")
    return total_loss / total_tokens


def save_sft_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: str | os.PathLike[str],
    metrics: dict[str, Any] | None = None,
) -> str:
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError("model must implement save_pretrained() for checkpoint saving.")
    model.save_pretrained(str(checkpoint_dir))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(checkpoint_dir))
    if metrics is not None:
        with (checkpoint_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    return str(checkpoint_dir)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_sft(
    model: torch.nn.Module,
    tokenizer: Any,
    train_data: Sequence[dict[str, Any]],
    validation_data: Sequence[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    *,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum_steps: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.03,
    max_length: int = 512,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    use_amp: bool = True,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run manual PyTorch SFT (no Trainer or TRL training loop)."""
    if not train_data:
        raise ValueError("train_data must contain at least one example.")
    if not validation_data:
        raise ValueError("validation_data must contain at least one example.")
    if min(epochs, batch_size, grad_accum_steps) < 1:
        raise ValueError("epochs, batch_size, and grad_accum_steps must be positive.")
    if learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("learning_rate and max_grad_norm must be positive; weight_decay non-negative.")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1).")

    from data_utils import set_seed

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(
        device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model.to(device=selected_device, dtype=torch.float32)
    prompts, responses = _formatted_rows(train_data, tokenizer)
    run_config = {
        "method": "manual_pytorch_sft",
        "model": getattr(getattr(model, "config", None), "_name_or_path", None),
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "epochs": epochs,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate": learning_rate,
        "scheduler": "cosine",
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "max_grad_norm": max_grad_norm,
        "precision": "fp16 autocast with fp32 weights"
        if use_amp and selected_device.type == "cuda"
        else "fp32",
        "sft_loss_reduction": "mean over all unmasked response tokens",
        "prompt_format": "tokenizer chat template when available; otherwise User/Assistant text",
        "seed": seed,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    train_rows = list(zip(prompts, responses))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: rows,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = math.ceil(len(loader) / grad_accum_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = bool(use_amp and selected_device.type == "cuda")
    scaler = _grad_scaler(amp_enabled)
    metrics_path = output_dir / "metrics.jsonl"
    best_validation = float("inf")
    best_epoch: int | None = None
    global_step = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        micro_batches = 0
        for batch_index, rows in enumerate(loader):
            batch_prompts, batch_responses = zip(*rows)
            group_start = (batch_index // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, len(loader) - group_start)
            with _autocast(selected_device, amp_enabled):
                loss = compute_sft_loss(
                    model, tokenizer, batch_prompts, batch_responses, max_length
                )
            scaler.scale(loss / group_size).backward()
            running_loss += float(loss.detach().item())
            micro_batches += 1
            should_update = (
                (batch_index + 1) % grad_accum_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        validation_loss = evaluate_sft(
            model,
            tokenizer,
            validation_data,
            max_length=max_length,
            batch_size=batch_size,
            use_amp=amp_enabled,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": running_loss / max(1, micro_batches),
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seed": seed,
        }
        _append_jsonl(metrics_path, epoch_metrics)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch + 1
            save_sft_checkpoint(model, tokenizer, output_dir / "checkpoint-best", epoch_metrics)

    final_metrics = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "final_epoch": epochs,
        "global_step": global_step,
        "seed": seed,
    }
    save_sft_checkpoint(model, tokenizer, output_dir / "checkpoint-final", final_metrics)
    return final_metrics
