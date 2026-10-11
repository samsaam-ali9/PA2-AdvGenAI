from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sft_training import _autocast, _grad_scaler, _model_device, build_sft_batch, save_sft_checkpoint


def build_dpo_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
    *,
    max_prompt_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses, and rejected_responses must align.")
    return {
        "chosen": build_sft_batch(
            tokenizer, prompts, chosen_responses, max_length,
            max_prompt_length=max_prompt_length,
        ),
        "rejected": build_sft_batch(
            tokenizer, prompts, rejected_responses, max_length,
            max_prompt_length=max_prompt_length,
        ),
    }


def _response_logprobs_from_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits
    shifted_logits = logits[:, :-1, :].contiguous().float()
    shifted_labels = batch["labels"][:, 1:].contiguous()
    if shifted_logits.size(1) == 0:
        return torch.zeros(
            batch["input_ids"].size(0),
            dtype=torch.float32,
            device=batch["input_ids"].device,
        )

    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logps.masked_fill(~mask, 0.0).sum(dim=-1)


def compute_response_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    batch = {key: value.to(_model_device(model)) for key, value in batch.items()}
    return _response_logprobs_from_batch(model, batch)


def compute_dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    
    if beta <= 0:
        raise ValueError("beta must be positive.")
    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("All log-probability inputs must be one-dimensional tensors.")
    if len({tuple(tensor.shape) for tensor in tensors}) != 1:
        raise ValueError("All log-probability inputs must have the same batch shape.")
    if policy_chosen_logps.numel() == 0:
        raise ValueError("The DPO loss requires at least one preference pair.")

    device = policy_chosen_logps.device
    policy_rejected_logps = policy_rejected_logps.to(device)
    reference_chosen_logps = reference_chosen_logps.to(device)
    reference_rejected_logps = reference_rejected_logps.to(device)
    preference_logits = beta * (
        (policy_chosen_logps - policy_rejected_logps)
        - (reference_chosen_logps - reference_rejected_logps)
    )
    return -F.logsigmoid(preference_logits).mean()


def compute_dpo_loss(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    beta: float,
    max_length: int,
    *,
    max_prompt_length: int | None = None,
) -> torch.Tensor:
    
    batch = build_dpo_batch(
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length,
        max_prompt_length=max_prompt_length,
    )
    policy_device = _model_device(policy_model)
    reference_device = _model_device(reference_model)
    policy_chosen = {key: value.to(policy_device) for key, value in batch["chosen"].items()}
    policy_rejected = {key: value.to(policy_device) for key, value in batch["rejected"].items()}
    reference_chosen = {key: value.to(reference_device) for key, value in batch["chosen"].items()}
    reference_rejected = {key: value.to(reference_device) for key, value in batch["rejected"].items()}

    policy_chosen_logps = _response_logprobs_from_batch(policy_model, policy_chosen)
    policy_rejected_logps = _response_logprobs_from_batch(policy_model, policy_rejected)
    reference_model.eval()
    with torch.no_grad():
        reference_chosen_logps = _response_logprobs_from_batch(reference_model, reference_chosen)
        reference_rejected_logps = _response_logprobs_from_batch(reference_model, reference_rejected)

    return compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_reward_metrics(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> dict[str, float]:
    
    if beta <= 0:
        raise ValueError("beta must be positive.")
    device = policy_chosen_logps.device
    reward_chosen = beta * (policy_chosen_logps - reference_chosen_logps.to(device))
    reward_rejected = beta * (policy_rejected_logps - reference_rejected_logps.to(device))
    margins = reward_chosen - reward_rejected
    return {
        "reward_accuracy": float((margins > 0).float().mean().item()),
        "reward_margin": float(margins.mean().item()),
        "reward_tie_rate": float((margins == 0).float().mean().item()),
    }


def _as_training_rows(
    rows: Sequence[dict[str, Any]], tokenizer: Any
) -> list[tuple[str, str, str]]:
    from data_utils import format_prompt

    formatted: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt", row.get("instruction"))
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if prompt is None or chosen is None or rejected is None:
            raise ValueError(f"DPO row {index} needs prompt/instruction, chosen, and rejected.")
        formatted.append((format_prompt(str(prompt), tokenizer), str(chosen), str(rejected)))
    return formatted


def _compute_pair_logps(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    chosen: Sequence[str],
    rejected: Sequence[str],
    max_length: int,
    max_prompt_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = build_dpo_batch(
        tokenizer, prompts, chosen, rejected, max_length,
        max_prompt_length=max_prompt_length,
    )
    policy_device = _model_device(policy_model)
    reference_device = _model_device(reference_model)
    policy_chosen = {key: value.to(policy_device) for key, value in batch["chosen"].items()}
    policy_rejected = {key: value.to(policy_device) for key, value in batch["rejected"].items()}
    reference_chosen = {key: value.to(reference_device) for key, value in batch["chosen"].items()}
    reference_rejected = {key: value.to(reference_device) for key, value in batch["rejected"].items()}

    policy_chosen_logps = _response_logprobs_from_batch(policy_model, policy_chosen)
    policy_rejected_logps = _response_logprobs_from_batch(policy_model, policy_rejected)
    with torch.no_grad():
        reference_chosen_logps = _response_logprobs_from_batch(reference_model, reference_chosen)
        reference_rejected_logps = _response_logprobs_from_batch(reference_model, reference_rejected)
    return policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps


def evaluate_dpo(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer: Any,
    validation_data: Sequence[dict[str, Any]],
    *,
    beta: float = 0.1,
    max_length: int = 512,
    max_prompt_length: int = 256,
    batch_size: int = 1,
    use_amp: bool = True,
) -> dict[str, float]:
    
    if not validation_data:
        raise ValueError("validation_data must contain at least one preference pair.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    was_training = policy_model.training
    policy_model.eval()
    reference_model.eval()
    device = _model_device(policy_model)
    rows = _as_training_rows(validation_data, tokenizer)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: batch)
    loss_sum = 0.0
    example_count = 0
    all_logps: list[list[torch.Tensor]] = [[], [], [], []]
    try:
        for batch_rows in loader:
            prompts, chosen, rejected = zip(*batch_rows)
            with torch.no_grad(), _autocast(device, use_amp and device.type == "cuda"):
                logps = _compute_pair_logps(
                    policy_model, reference_model, tokenizer,
                    prompts, chosen, rejected, max_length, max_prompt_length,
                )
                loss = compute_dpo_loss_from_logps(*logps, beta)
            count = len(batch_rows)
            loss_sum += float(loss.item()) * count
            example_count += count
            for stored, values in zip(all_logps, logps):
                stored.append(values.detach())
    finally:
        if was_training:
            policy_model.train()

    metrics = compute_reward_metrics(*(torch.cat(values) for values in all_logps), beta)
    metrics["dpo_loss"] = loss_sum / example_count
    return metrics


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_dpo(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer: Any,
    train_data: Sequence[dict[str, Any]],
    validation_data: Sequence[dict[str, Any]],
    output_dir: str | os.PathLike[str],
    *,
    beta: float = 0.1,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 32,
    learning_rate: float = 1e-6,
    max_length: int = 512,
    max_prompt_length: int = 256,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    use_amp: bool = True,
    policy_device: str | torch.device | None = None,
    reference_device: str | torch.device | None = None,
    reference_fp16: bool = True,
) -> dict[str, Any]:
    
    if not train_data:
        raise ValueError("train_data must contain at least one preference pair.")
    if not validation_data:
        raise ValueError("validation_data must contain at least one preference pair.")
    if min(epochs, batch_size, grad_accum_steps) < 1:
        raise ValueError("epochs, batch_size, and grad_accum_steps must be positive.")
    if beta <= 0 or learning_rate <= 0 or max_grad_norm <= 0:
        raise ValueError("beta, learning_rate, and max_grad_norm must be positive.")
    if max_prompt_length < 1 or max_length < 2:
        raise ValueError("max_length must be at least 2 and max_prompt_length must be positive.")
    if policy_model is reference_model:
        raise ValueError("policy_model and reference_model must be separate model instances.")

    from data_utils import set_seed

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if policy_device is None:
        policy_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if reference_device is None:
        reference_device = (
            "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else policy_device
        )
    policy_device = torch.device(policy_device)
    reference_device = torch.device(reference_device)

    policy_model.to(device=policy_device, dtype=torch.float32)
    reference_dtype = torch.float16 if reference_fp16 and reference_device.type == "cuda" else torch.float32
    reference_model.to(device=reference_device, dtype=reference_dtype)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    formatted = _as_training_rows(train_data, tokenizer)
    run_config = {
        "method": "manual_pytorch_dpo",
        "policy_model": getattr(getattr(policy_model, "config", None), "_name_or_path", None),
        "reference_model": getattr(getattr(reference_model, "config", None), "_name_or_path", None),
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "beta": beta,
        "epochs": epochs,
        "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate": learning_rate,
        "scheduler": "constant",
        "max_length": max_length,
        "max_prompt_length": max_prompt_length,
        "max_grad_norm": max_grad_norm,
        "policy_precision": "fp16 autocast with fp32 weights"
        if use_amp and policy_device.type == "cuda" else "fp32",
        "reference_precision": str(reference_dtype),
        "policy_device": str(policy_device),
        "reference_device": str(reference_device),
        "checkpoint_selection": "highest validation reward accuracy, then lowest validation DPO loss",
        "prompt_format": "tokenizer chat template when available; otherwise User/Assistant text",
        "seed": seed,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        formatted,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: batch,
    )
    trainable_parameters = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The policy model has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=0.0)
    amp_enabled = bool(use_amp and policy_device.type == "cuda")
    scaler = _grad_scaler(amp_enabled)
    metrics_path = output_dir / "metrics.jsonl"
    best_accuracy = -1.0
    best_validation_loss = float("inf")
    best_margin = float("-inf")
    best_epoch: int | None = None
    global_step = 0

    for epoch in range(epochs):
        policy_model.train()
        reference_model.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        micro_batches = 0
        for batch_index, batch_rows in enumerate(loader):
            prompts, chosen, rejected = zip(*batch_rows)
            group_start = (batch_index // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, len(loader) - group_start)
            with _autocast(policy_device, amp_enabled):
                loss = compute_dpo_loss(
                    policy_model, reference_model, tokenizer,
                    prompts, chosen, rejected, beta, max_length,
                    max_prompt_length=max_prompt_length,
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
                global_step += 1

        validation_metrics = evaluate_dpo(
            policy_model,
            reference_model,
            tokenizer,
            validation_data,
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            batch_size=batch_size,
            use_amp=amp_enabled,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": running_loss / max(1, micro_batches),
            "validation_dpo_loss": validation_metrics["dpo_loss"],
            "validation_reward_accuracy": validation_metrics["reward_accuracy"],
            "validation_reward_margin": validation_metrics["reward_margin"],
            "validation_reward_tie_rate": validation_metrics["reward_tie_rate"],
            "beta": beta,
            "seed": seed,
        }
        _append_jsonl(metrics_path, epoch_metrics)

        accuracy = validation_metrics["reward_accuracy"]
        validation_loss = validation_metrics["dpo_loss"]
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and validation_loss < best_validation_loss
        ):
            best_accuracy = accuracy
            best_validation_loss = validation_loss
            best_margin = validation_metrics["reward_margin"]
            best_epoch = epoch + 1
            save_sft_checkpoint(policy_model, tokenizer, output_dir / "checkpoint-best", epoch_metrics)

    final_metrics = {
        "best_epoch": best_epoch,
        "best_validation_reward_accuracy": best_accuracy,
        "best_validation_reward_margin": best_margin,
        "best_validation_dpo_loss": best_validation_loss,
        "final_epoch": epochs,
        "global_step": global_step,
        "beta": beta,
        "seed": seed,
    }
    save_sft_checkpoint(policy_model, tokenizer, output_dir / "checkpoint-final", final_metrics)
    return final_metrics
