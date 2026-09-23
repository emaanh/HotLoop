"""Pick models by one objective rule: top-N Hugging Face models by downloads,
deduplicated by architecture signature. No hand-picked model lists."""

import json

# Config fields that change the math a model runs. Two checkpoints with the
# same signature (fine-tunes, quantized re-uploads, ...) produce identical tasks.
SIGNATURE_FIELDS = [
    "model_type", "hidden_size", "intermediate_size", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "num_local_experts",
    "num_experts", "num_experts_per_tok", "moe_intermediate_size",
    "hidden_act", "tie_word_embeddings",
]


def signature(cfg: dict) -> str:
    return json.dumps({k: cfg.get(k) for k in SIGNATURE_FIELDS}, sort_keys=True)


MIN_PARAMS = 100e6  # excludes tiny test fixtures (e.g. trl-internal-testing/*)


def approx_params(cfg) -> float:
    """From a transformers config object (its attribute_map normalises n_embd, n_layer, ...)."""
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    h = getattr(cfg, "hidden_size", 0) or 0
    L = getattr(cfg, "num_hidden_layers", 0) or 0
    v = getattr(cfg, "vocab_size", 0) or 0
    inter = getattr(cfg, "intermediate_size", None) or 4 * h
    return L * (4 * h * h + 3 * h * inter) + v * h


def select_models(n: int, pipeline_tag: str = "text-generation", scan: int = 300) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    api = HfApi()
    seen, picked = set(), []
    for m in api.list_models(pipeline_tag=pipeline_tag, sort="downloads", limit=scan):
        if len(picked) >= n:
            break
        try:
            with open(hf_hub_download(m.id, "config.json")) as f:
                raw = json.load(f)
        except Exception:
            continue  # gated without access, no config, etc.
        if raw.get("quantization_config") or raw.get("auto_map"):
            continue  # quantized re-upload or needs remote code
        if raw.get("model_type") not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            continue
        try:
            cfg = AutoConfig.from_pretrained(m.id)
        except Exception:
            continue
        if approx_params(cfg) < MIN_PARAMS:
            continue
        sig = signature(raw)
        if sig in seen:
            continue
        seen.add(sig)
        picked.append({"model_id": m.id, "downloads": m.downloads, "signature": sig})
    return picked
