"""Model catalog: real Hugging Face model metadata and compatibility checks.

Compatibility statements are limited to what the installed libraries and the
model's own metadata actually support — nothing is invented when unknown.
"""

from __future__ import annotations

from typing import Any

import httpx

from .hardware import inspect_dependencies

HF_API = "https://huggingface.co/api"


class ModelCatalogError(Exception):
    def __init__(self, user_message: str, detail: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


LORA_SUPPORTED_ARCHS: dict[str, dict[str, Any]] = {
    "LlamaForCausalLM": {"lora": True},
    "MistralForCausalLM": {"lora": True},
    "GemmaForCausalLM": {"lora": True},
    "Gemma2ForCausalLM": {"lora": True},
    "Qwen2ForCausalLM": {"lora": True},
    "GPT2LMHeadModel": {"lora": True},
    "SmolLM3ForCausalLM": {"lora": True},
    "OPTForCausalLM": {"lora": True},
    "BloomForCausalLM": {"lora": True},
    "TinyRandomForCausalLM": {"lora": True},
}


def _hf_arch(cfg: dict[str, Any]) -> str | None:
    for key in ("architectures",):
        archs = cfg.get(key)
        if isinstance(archs, list) and archs:
            return archs[0]
    return None


def inspect_model(repo_id: str, revision: str = "main") -> dict[str, Any]:
    """Fetch real model metadata: config, size, license, tokenizer info."""
    try:
        cfg_resp = httpx.get(
            f"{HF_API}/models/{repo_id}/revision/{revision}/config.json", timeout=30.0
        )
        info_resp = httpx.get(f"{HF_API}/models/{repo_id}?blobs=true&revision={revision}", timeout=30.0)
    except httpx.ConnectError as exc:
        raise ModelCatalogError("Hugging Face'e bağlanılamadı (ağ erişimi gerekli).", str(exc)) from exc
    except httpx.TimeoutException as exc:
        raise ModelCatalogError("Hugging Face isteği zaman aşımına uğradı.", str(exc)) from exc
    if cfg_resp.status_code in (401, 403):
        raise ModelCatalogError(
            f"{repo_id} kısıtlı/gated bir model; erişim kontrolü aşılmaz. Yetkili belgeler gerekli."
        )
    if cfg_resp.status_code == 404:
        raise ModelCatalogError(f"Model bulunamadı veya config.json dosyası yok: {repo_id}")
    cfg_resp.raise_for_status()
    info_resp.raise_for_status()
    cfg = cfg_resp.json() or {}
    info = info_resp.json() or {}
    tags = info.get("tags") or []
    license_name = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)

    total_bytes = 0
    safetensors_bytes = 0
    for s in info.get("siblings") or []:
        name = s.get("rfilename") or ""
        size = s.get("size")
        if size is None:
            # blobs=true should provide real sizes; if not, unknown stays unknown.
            continue
        total_bytes += size
        if name.endswith(".safetensors"):
            safetensors_bytes += size

    model_config = info.get("config") or {}
    architectures = cfg.get("architectures") or model_config.get("architectures")
    arch = architectures[0] if isinstance(architectures, list) and architectures else None
    compat = compatibility_for_arch(arch, cfg)
    return {
        "id": info.get("modelId") or repo_id,
        "revision": revision,
        "architectures": architectures or None,
        "architecture": arch,
        "license": license_name,
        "tokenizer_class": (cfg.get("tokenizer_class") or model_config.get("tokenizer_class")),
        "chat_template_present": bool(cfg.get("chat_template") or model_config.get("chat_template")),
        "hidden_size": cfg.get("hidden_size"),
        "num_parameters_estimate": None,  # never guessed; only measured where computable
        "total_disk_bytes": total_bytes or None,
        "safetensors_bytes": safetensors_bytes or None,
        "gated": bool(info.get("gated")),
        "private": bool(info.get("private")),
        "compatibility": compat,
        "provenance": "huggingface_hub_api",
    }


def compatibility_for_arch(arch: str | None, cfg: dict[str, Any]) -> dict[str, Any]:
    deps = inspect_dependencies()
    available = bool(deps.get("torch") and deps.get("transformers") and deps.get("peft"))
    ctx = bool(cfg.get("use_cache", True)) if isinstance(cfg.get("use_cache"), bool) else None
    lora_ok = arch in LORA_SUPPORTED_ARCHS
    modes: dict[str, Any] = {
        "configuration_only": {"available": available},
        "lora": {
            "available": available and lora_ok,
            "note": None if lora_ok else (
                f"Mimari {arch} varsayılan LoRA destek listesinde değil; PEFT hedef modül "
                "auto-mapping bu mimari için uyarlanmamış."
            ),
        },
        "qlora": {
            "available": False,
            "note": (
                "QLoRA, bitsandbytes + CUDA gerektirir; bu ortamda kullanılamaz."
                if not deps.get("bitsandbytes")
                else "bitsandbytes kurulu; GPU gerekli. Kullanımdan önce ön kontrol (preflight) yapılır."
            ),
        },
        "full_finetune": {
            "available": available and lora_ok,
            "note": "Tam ince ayar bellek gereksinimi yüksek; preflight uyarır.",
        },
        "pretrain": {
            "available": False,
            "note": "Sıfırdan önedirme bu sürümde uygulanmadı.",
        },
    }
    return {"trained_engine": "transformers+peft", "architecture": arch, "modes": modes, "use_cache": ctx}
