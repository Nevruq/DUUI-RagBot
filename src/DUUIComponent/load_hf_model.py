#!/usr/bin/env python3
"""
hf_model_inspect.py

Inspect a Hugging Face model repo (and optional revision) and export a JSON
with the key facts an LLM needs to generate correct code + typesystem.

Usage:
  python hf_model_inspect.py --model distilbert/distilbert-base-uncased-finetuned-sst-2-english --out model_info.json
  python hf_model_inspect.py --model debajyotimaz/codemix_hate --revision b07d73f... --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Transformers is required
try:
    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
    )
except Exception as e:
    print("ERROR: This script requires 'transformers'. Install with: pip install transformers", file=sys.stderr)
    raise

# Torch is optional but recommended for probing forward pass / device info
try:
    import torch
except Exception:
    torch = None  # type: ignore


@dataclass
class HFModelInfo:
    model_id: str
    revision: Optional[str] = None

    transformers_version: str = ""
    torch_available: bool = False
    device_used: str = "cpu"

    # Config basics
    model_type: Optional[str] = None
    architectures: List[str] = field(default_factory=list)
    num_labels: Optional[int] = None
    id2label: Dict[str, str] = field(default_factory=dict)     # string keys for JSON safety
    label2id: Dict[str, int] = field(default_factory=dict)
    problem_type: Optional[str] = None

    # Tokenizer basics
    tokenizer_class: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    vocab_size: Optional[int] = None
    model_max_length: Optional[int] = None
    pad_token: Optional[str] = None
    unk_token: Optional[str] = None
    cls_token: Optional[str] = None
    sep_token: Optional[str] = None
    mask_token: Optional[str] = None

    # Heuristics / hints for LLM
    inferred_task: Optional[str] = None
    output_semantics_hint: Optional[str] = None
    recommended_postprocess: Optional[str] = None

    # Forward probe (best effort)
    probe: Dict[str, Any] = field(default_factory=dict)

    # Any warnings/errors encountered
    warnings: List[str] = field(default_factory=list)


def _safe_int_keys(d: Dict[Any, Any]) -> Dict[str, Any]:
    """Convert dict keys to strings for JSON stability."""
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        out[str(k)] = v
    return out


def infer_task_from_config(cfg: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Best-effort task inference + semantics + postprocess based on config.
    Returns: (task, semantics_hint, postprocess)
    """
    archs = [a.lower() for a in (getattr(cfg, "architectures", None) or [])]
    num_labels = getattr(cfg, "num_labels", None)
    problem_type = getattr(cfg, "problem_type", None)

    # Common arch patterns
    if any("forsequenceclassification" in a for a in archs):
        semantics = "Sequence-level classification: output logits shape [batch, num_labels]."
        if problem_type == "multi_label_classification":
            post = "sigmoid on logits per label; threshold or top-k."
        else:
            post = "softmax over logits; pick argmax/top-k."
        return ("text-classification", semantics, post)

    if any("fortokenclassification" in a for a in archs):
        semantics = "Token-level classification: output logits shape [batch, seq_len, num_labels]."
        post = "softmax per token (or argmax) then map to labels; align with tokens/offsets."
        return ("token-classification", semantics, post)

    if any("forquestionanswering" in a for a in archs):
        semantics = "Extractive QA: outputs start_logits and end_logits shape [batch, seq_len]."
        post = "argmax (or beam) over start/end; decode span via tokenizer offsets."
        return ("question-answering", semantics, post)

    if any("forseq2seqlm" in a for a in archs) or getattr(cfg, "is_encoder_decoder", False):
        semantics = "Seq2Seq generation: use model.generate(); outputs token ids -> decoded text."
        post = "use generate() with max_new_tokens; decode with tokenizer."
        return ("text2text-generation", semantics, post)

    if any("forcausallm" in a for a in archs):
        semantics = "Causal LM generation: use model.generate(); outputs token ids -> decoded text."
        post = "use generate(); decode with tokenizer."
        return ("text-generation", semantics, post)

    # Heuristic fallback: if num_labels exists and small => likely classification
    if isinstance(num_labels, int) and num_labels > 0 and num_labels <= 10:
        semantics = "Likely classification head: output logits [batch, num_labels]. Verify by probing."
        post = "softmax over logits (or sigmoid if multi-label)."
        return ("unknown (likely classification)", semantics, post)

    return (None, None, None)


def choose_model_loader(task: Optional[str]):
    """
    Pick a sensible AutoModel* class to load for probing.
    """
    if task == "text-classification" or (task and "classification" in task):
        return AutoModelForSequenceClassification
    if task == "token-classification":
        return AutoModelForTokenClassification
    if task == "question-answering":
        return AutoModelForQuestionAnswering
    if task == "text2text-generation":
        return AutoModelForSeq2SeqLM
    if task == "text-generation":
        return AutoModelForCausalLM
    return AutoModel


def probe_forward(
    model: Any,
    tokenizer: Any,
    device: str,
    task: Optional[str],
    warnings: List[str],
) -> Dict[str, Any]:
    """
    Run a tiny dummy forward pass (best effort) and summarize output structure.
    """
    out: Dict[str, Any] = {}

    if torch is None:
        warnings.append("torch not available; skipping forward probe.")
        return out

    try:
        # Prepare minimal dummy inputs depending on task
        if task == "question-answering":
            inputs = tokenizer(
                "What is the capital of France?",
                "France's capital is Paris.",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
        else:
            inputs = tokenizer(
                ["Hello world!", "This is a test."],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )

        # Move inputs to device if possible
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        # outputs can be ModelOutput (dict-like) or tuple
        if hasattr(outputs, "to_tuple"):
            tup = outputs.to_tuple()
            out["output_is_modeloutput"] = True
            out["output_tuple_len"] = len(tup)
        else:
            out["output_is_modeloutput"] = False

        # Try to summarize common fields
        def shape_of(x):
            try:
                return list(x.shape)
            except Exception:
                return None

        # Access as dict-like when possible
        fields = {}
        for key in ["logits", "start_logits", "end_logits", "loss"]:
            if hasattr(outputs, key):
                val = getattr(outputs, key)
                fields[key] = {"shape": shape_of(val), "dtype": str(getattr(val, "dtype", None))}
        out["fields"] = fields

        # Also list available keys for ModelOutput
        if hasattr(outputs, "keys"):
            out["available_keys"] = list(outputs.keys())

        return out

    except Exception as e:
        warnings.append(f"Forward probe failed: {type(e).__name__}: {e}")
        return out


def inspect_model(model_id: str, revision: Optional[str], device: str, local_files_only: bool) -> HFModelInfo:
    info = HFModelInfo(
        model_id=model_id,
        revision=revision,
        transformers_version=getattr(transformers, "__version__", "unknown"),
        torch_available=torch is not None,
        device_used=device if (torch is not None and device != "cpu") else "cpu",
    )

    # Helpful environment hint
    if local_files_only:
        info.warnings.append("local_files_only=True: will not download; requires model/tokenizer in cache.")

    # Load config
    try:
        cfg = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as e:
        info.warnings.append(f"Failed to load config: {type(e).__name__}: {e}")
        return info

    info.model_type = getattr(cfg, "model_type", None)
    info.architectures = list(getattr(cfg, "architectures", None) or [])
    info.num_labels = getattr(cfg, "num_labels", None)
    info.problem_type = getattr(cfg, "problem_type", None)

    # id2label / label2id
    id2label = getattr(cfg, "id2label", None) or {}
    label2id = getattr(cfg, "label2id", None) or {}

    # Ensure JSON-safe key types
    info.id2label = _safe_int_keys(id2label)
    # label2id keys are strings already, but values may be non-int
    try:
        info.label2id = {str(k): int(v) for k, v in label2id.items()}
    except Exception:
        info.label2id = {str(k): v for k, v in label2id.items()}

    # Infer task & hints
    task, semantics, post = infer_task_from_config(cfg)
    info.inferred_task = task
    info.output_semantics_hint = semantics
    info.recommended_postprocess = post

    # Load tokenizer (best effort)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            use_fast=True,
        )
        info.tokenizer_class = tokenizer.__class__.__name__
        info.tokenizer_name_or_path = getattr(tokenizer, "name_or_path", None)
        info.vocab_size = getattr(tokenizer, "vocab_size", None)
        info.model_max_length = getattr(tokenizer, "model_max_length", None)

        info.pad_token = getattr(tokenizer, "pad_token", None)
        info.unk_token = getattr(tokenizer, "unk_token", None)
        info.cls_token = getattr(tokenizer, "cls_token", None)
        info.sep_token = getattr(tokenizer, "sep_token", None)
        info.mask_token = getattr(tokenizer, "mask_token", None)
    except Exception as e:
        info.warnings.append(f"Failed to load tokenizer: {type(e).__name__}: {e}")

    # Load model for probing (best effort)
    if tokenizer is not None:
        model_cls = choose_model_loader(info.inferred_task)
        try:
            model = model_cls.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=local_files_only,
            )
            if torch is not None:
                # Move to device if possible
                if device != "cpu":
                    try:
                        model = model.to(device)
                        info.device_used = device
                    except Exception as e:
                        info.warnings.append(f"Could not move model to device '{device}': {type(e).__name__}: {e}")
                        info.device_used = "cpu"

            info.probe = probe_forward(model, tokenizer, info.device_used, info.inferred_task, info.warnings)
        except Exception as e:
            info.warnings.append(f"Failed to load model for probing: {type(e).__name__}: {e}")

    # Extra sanity warnings
    if info.inferred_task and "classification" in info.inferred_task and not info.id2label:
        info.warnings.append("Classification-like model but config.id2label is empty; you may need a manual label mapping.")

    return info

def inspect_hf_model(
    model_id: str,
    revision: str | None = None,
    device: str = "cpu",
    local_files_only: bool = False,
    return_json: bool = False,
):
    """
    Programmatic API to inspect a Hugging Face model.

    Returns:
      - dict (default)
      - JSON string if return_json=True
    """
    info = inspect_model(
        model_id=model_id,
        revision=revision,
        device=device,
        local_files_only=local_files_only,
    )

    payload = asdict(info)

    if return_json:
        import json
        return json.dumps(payload, indent=2, ensure_ascii=False)

    return payload

if __name__ == "__main__":


    info = inspect_hf_model(
    model_id="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(info["inferred_task"])
    print(info["id2label"])