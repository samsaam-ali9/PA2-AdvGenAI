"""Dataset loading and prompt formatting for the alignment assignment."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_SFT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_HH_DATASET = "Anthropic/hh-rlhf"
DEFAULT_ULTRAFEEDBACK_DATASET = "HuggingFaceH4/ultrafeedback_binarized"




def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_prompt(instruction: str, tokenizer: Any | None = None) -> str:
    instruction = str(instruction).strip()
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, TypeError, ValueError):
            pass
    return f"User: {instruction}\nAssistant:"


def _load_hf_split(
    dataset_name: str,
    split: str,
    config_name: str | None = None,
    streaming: bool = True,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the assignment requirements before loading Hugging Face datasets."
        ) from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if config_name is not None:
        kwargs["name"] = config_name
    return load_dataset(dataset_name, **kwargs)


def _shuffled(source: Iterable[dict[str, Any]], seed: int) -> Iterable[dict[str, Any]]:
    shuffle = getattr(source, "shuffle", None)
    if callable(shuffle):
        try:
            return shuffle(seed=seed, buffer_size=10_000)
        except TypeError:
            try:
                return shuffle(seed=seed)
            except TypeError:
                pass

    rows = list(source)
    random.Random(seed).shuffle(rows)
    return rows


def _take_examples(
    source: Iterable[dict[str, Any]],
    converter: Callable[[dict[str, Any]], dict[str, str] | None],
    count: int,
    seed: int,
    split_name: str,
) -> list[dict[str, str]]:
    if count < 0:
        raise ValueError("Example counts must be non-negative.")
    if count == 0:
        return []

    examples: list[dict[str, str]] = []
    for raw in _shuffled(source, seed):
        example = converter(raw)
        if example is not None:
            examples.append(example)
            if len(examples) == count:
                break

    if len(examples) != count:
        raise ValueError(
            f"Requested {count} usable examples from '{split_name}', but found "
            f"{len(examples)}. Check the dataset split and single-turn filters."
        )
    return examples


def _messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, (list, tuple)):
        return None

    result = []
    for message in value:
        if not isinstance(message, dict):
            return None
        role = message.get("role", message.get("from", message.get("speaker")))
        content = message.get("content", message.get("value", message.get("text")))
        if role is None or content is None:
            return None

        role = str(role).lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "bot"}:
            role = "assistant"
        else:
            return None

        result.append({"role": role, "content": str(content).strip()})
    return result or None


def _single_sft_example(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = row.get("instruction", row.get("prompt"))
    response = row.get("response", row.get("output", row.get("completion")))
    if prompt is not None and response is not None:
        prompt, response = str(prompt).strip(), str(response).strip()
        if prompt and response:
            return {"instruction": prompt, "response": response}

    for field in ("messages", "conversations", "conversation"):
        dialogue = _messages(row.get(field))
        if dialogue:
            for i in range(len(dialogue) - 1):
                if (
                    dialogue[i]["role"] == "user"
                    and dialogue[i + 1]["role"] == "assistant"
                    and dialogue[i]["content"]
                    and dialogue[i + 1]["content"]
                ):
                    return {
                        "instruction": dialogue[i]["content"],
                        "response": dialogue[i + 1]["content"],
                    }
    return None


def load_sft_data(
    train_size: int = 30_000,
    validation_size: int = 1_000,
    dataset_name: str = DEFAULT_SFT_DATASET,
    config_name: str | None = None,
    train_split: str = "train_sft",
    validation_split: str = "test_sft",
    seed: int = 42,
    streaming: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = _load_hf_split(dataset_name, train_split, config_name, streaming)
    validation = _load_hf_split(dataset_name, validation_split, config_name, streaming)
    return (
        _take_examples(train, _single_sft_example, train_size, seed, train_split),
        _take_examples(
            validation,
            _single_sft_example,
            validation_size,
            seed + 1,
            validation_split,
        ),
    )


def _parse_hh_single_turn(text: Any) -> tuple[str, str] | None:
    if not isinstance(text, str):
        return None

    markers = list(re.finditer(r"(?m)^(Human|Assistant): ", text))
    assistants = [m for m in markers if m.group(1) == "Assistant"]

    if not assistants:
        return None

    last = assistants[-1]
    prompt = text[:last.start()].strip()
    response = text[last.end():].strip()

    if not prompt or not response:
        return None
    if not re.search(r"(?m)^Human: ", prompt):
        return None

    return prompt, response

def _hh_example(row: dict[str, Any]) -> dict[str, str] | None:
    chosen = _parse_hh_single_turn(row.get("chosen"))
    rejected = _parse_hh_single_turn(row.get("rejected"))
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    return {"prompt": chosen[0], "chosen": chosen[1], "rejected": rejected[1]}


def load_hh_preferences(
    train_size: int = 15_000,
    validation_size: int = 1_000,
    dataset_name: str = DEFAULT_HH_DATASET,
    config_name: str | None = None,
    train_split: str = "train",
    validation_split: str = "test",
    seed: int = 42,
    streaming: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = _load_hf_split(dataset_name, train_split, config_name, streaming)
    validation = _load_hf_split(dataset_name, validation_split, config_name, streaming)
    return (
        _take_examples(train, _hh_example, train_size, seed, train_split),
        _take_examples(
            validation,
            _hh_example,
            validation_size,
            seed + 1,
            validation_split,
        ),
    )


def _as_single_prompt(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    dialogue = _messages(value)
    if dialogue and len(dialogue) == 1 and dialogue[0]["role"] == "user":
        return dialogue[0]["content"] or None
    return None


def _as_single_response(value: Any, prompt: str) -> str | None:
    if isinstance(value, str):
        return value.strip() or None

    dialogue = _messages(value)
    if not dialogue:
        return None
    if (
        len(dialogue) == 2
        and dialogue[0]["role"] == "user"
        and dialogue[0]["content"] == prompt
        and dialogue[1]["role"] == "assistant"
    ):
        return dialogue[1]["content"] or None
    if len(dialogue) == 1 and dialogue[0]["role"] == "assistant":
        return dialogue[0]["content"] or None
    return None


def _ultrafeedback_example(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = _as_single_prompt(row.get("prompt"))
    if prompt is None:
        return None

    chosen = _as_single_response(row.get("chosen"), prompt)
    rejected = _as_single_response(row.get("rejected"), prompt)
    if chosen is None or rejected is None:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def load_ultrafeedback_data(
    train_size: int = 15_000,
    validation_size: int = 1_000,
    dataset_name: str = DEFAULT_ULTRAFEEDBACK_DATASET,
    config_name: str | None = None,
    train_split: str = "train_prefs",
    validation_split: str = "test_prefs",
    seed: int = 42,
    streaming: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train = _load_hf_split(dataset_name, train_split, config_name, streaming)
    validation = _load_hf_split(dataset_name, validation_split, config_name, streaming)
    return (
        _take_examples(train, _ultrafeedback_example, train_size, seed, train_split),
        _take_examples(
            validation,
            _ultrafeedback_example,
            validation_size,
            seed + 1,
            validation_split,
        ),
    )


def load_persona_data(
    path: str | Path | None = None,
) -> list[dict[str, str]]:
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "persona_lora" / "ramsay_style_preferences.json"

    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("data", data.get("examples", data.get("preferences")))
    if not isinstance(data, list):
        raise ValueError("The persona JSON must contain a list of preference examples.")

    examples = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Persona example {index} is not a JSON object.")
        values = [row.get("prompt"), row.get("chosen"), row.get("rejected")]
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"Persona example {index} needs non-empty prompt, chosen, and rejected fields.")
        examples.append(
            {
                "prompt": values[0].strip(),
                "chosen": values[1].strip(),
                "rejected": values[2].strip(),
            }
        )
    return examples
