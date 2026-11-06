"""Frozen Qwen3.5-9B text-only runtime identity and loading helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_MODEL_PATH = Path(os.environ.get("MESA_QA_MODEL_PATH", Path(__file__).resolve().parent / "models" / "Qwen3.5-9B"))
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}
RUNTIME_VERSION = "qwen35.text_only.v1"
PARSER_VERSION = "model_agnostic.generated_suffix.v1"


def load_text_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the frozen BF16 text-only Qwen3.5 runtime."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    path = Path(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(path),
        dtype=DTYPE,
        device_map="auto",
        attn_implementation=ATTENTION_IMPLEMENTATION,
        local_files_only=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    return model, processor


def render_text_chat(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **CHAT_TEMPLATE_KWARGS,
    )


def generated_suffix(input_ids: Any, output_ids: Any) -> list[Any]:
    """Extract newly generated IDs without depending on padding direction."""
    return [out[len(inp):] for inp, out in zip(input_ids, output_ids)]


def model_identity(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path).resolve()
    config = path / "config.json"
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_path": str(path),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "runtime_version": RUNTIME_VERSION,
        "dtype": DTYPE_NAME,
        "device_map": "auto",
        "attention_implementation": ATTENTION_IMPLEMENTATION,
    }


def cache_identity(model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
    return hashlib.sha256(json.dumps(model_identity(model_path), sort_keys=True).encode()).hexdigest()
