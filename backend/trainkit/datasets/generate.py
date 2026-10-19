"""Synthetic dataset generation using the connected Ollama model.

Every generated example is validated against supported schemas; malformed or
empty examples are rejected and duplicates detected. Only real counts are
reported — the dataset is never padded to hit the requested target.
"""

from __future__ import annotations

import re
import sys
import json
from dataclasses import dataclass, field
from typing import Any

from ..ollama.client import OllamaClient
from .schemas import examples_hash, normalize_example

SCHEMA_HINT = """[
  {"instruction": "kullanıcı talebi", "input": "varsa ek bağlam", "output": "beklenen yanıt"}
]"""

CHAT_SCHEMA_HINT = """[
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
]"""


@dataclass
class GenerationOptions:
    target_count: int = 50
    batch_size: int = 5
    temperature: float = 0.8
    schema_type: str = "instruction"  # instruction | chat
    seed: int | None = None


@dataclass
class GenerationResult:
    examples: list[dict[str, Any]] = field(default_factory=list)
    generated: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    batches: int = 0
    failed_batches: list[str] = field(default_factory=list)
    target: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "generated": self.generated,
            "accepted": len(self.examples),
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "batches": self.batches,
            "failed_batches": self.failed_batches,
            "count": len(self.examples),
        }


def _build_prompt(spec: dict[str, Any], options: GenerationOptions, offset: int, want: int) -> str:
    languages = " ve ".join(spec.get("languages") or ["Türkçe"])
    return (
        "Egitim icin sentetik ornek uretilecek modelin ozellikleri:\n"
        f"- Isim: {spec.get('target_model_name') or 'proje modeli'}\n"
        f"- Amac: {spec.get('purpose') or '-'}\n"
        f"- Hedef kullanicilar: {spec.get('intended_users') or '-'}\n"
        f"- Diller: {languages}\n"
        f"- Kisilik/ton: {spec.get('personality') or '-'}\n"
        f"- Istenen davranislar: {'; '.join(spec.get('desired_behavior') or [])}\n"
        f"- Istenmeyen davranislar: {'; '.join(spec.get('unwanted_behavior') or [])}\n"
        f"- Ornek kullanici girdileri: {'; '.join(spec.get('example_inputs') or [])}\n"
        f"- Sistem talimatlari: {spec.get('system_instructions') or '-'}\n\n"
        f"index {offset} sonrasindayiz; su anda TAM OLARAK {want} adet YENI ve "
        "BENZERSIZ egitim ornegi uret. Her ornek benzersiz olsun; oncekileri tekrar etme. "
        "Cikti olarak yalnizca JSON dizisi dondur."
    )


def _parse_batch(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array of raw examples from model output."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model ciktisinda JSON dizi bulunamadi")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("JSON dizi bekleniyordu")
    return parsed


def generate_dataset(
    client: OllamaClient,
    model: str,
    spec: dict[str, Any],
    options: GenerationOptions,
    *,
    on_batch: Any = None,
    cancel: Any = None,
) -> GenerationResult:
    """Generate real examples via Ollama. Returns the ACTUAL counts.

    on_batch(batch_index, info) observes progress. Bounded loop: at most 500
    batches are attempted under any circumstances.
    """
    import json

    result = GenerationResult(target=options.target_count)
    seen: set[str] = set()
    schema_fields = (
        "instruction/input/output" if options.schema_type == "instruction" else "messages"
    )

    batch_index = 0
    max_batches = 500
    while len(result.examples) < options.target_count and batch_index < max_batches:
        if cancel is not None and getattr(cancel, "is_set", None) and cancel.is_set():
            break
        batch_index += 1
        remaining = options.target_count - len(result.examples)
        want = min(remaining, options.batch_size)
        batch_options: dict[str, Any] = {
            "temperature": options.temperature,
            "num_predict": 4096,
        }
        if options.seed is not None:
            batch_options["seed"] = options.seed + batch_index

        prompt = _build_prompt(spec, options, len(result.examples), want)
        try:
            raw = client.generate(
                model,
                prompt,
                system=(
                    "Sen bir egitim verisi uretme motorusun. YALNIZCA gecerli JSON dizi dondur; "
                    f"dizinin her ogesi su alanlari icersin: {schema_fields}. "
                    "Baska hicbir metin ekleme."
                ),
                format="json",
                options=batch_options,
            )
        except Exception as exc:
            from ..ollama import OllamaError

            message = exc.user_message if isinstance(exc, OllamaError) else str(exc)
            result.failed_batches.append(f"batch {batch_index}: {message}")
            if on_batch:
                on_batch(batch_index, {"ok": False, "error": message})
            continue

        try:
            raw_examples = _parse_batch(raw)
        except (ValueError, Exception) as exc:  # noqa: B014 - want a single clear path here
            result.failed_batches.append(f"batch {batch_index}: {exc}")
            if on_batch:
                on_batch(batch_index, {"ok": False, "error": str(exc)})
            continue

        batch_generated = 0
        for item in raw_examples:
            batch_generated += 1
            result.generated += 1
            normalized = normalize_example(item, index=len(result.examples), source_id="ollama")
            if normalized is None:
                result.rejected += 1
                continue
            kind, example = normalized
            if kind != options.schema_type:
                result.rejected += 1
                continue
            h = examples_hash(example)
            if h in seen:
                result.duplicates += 1
                continue
            seen.add(h)
            example["_generated_with"] = {
                "model": model,
                "temperature": options.temperature,
                "batch": batch_index,
            }
            result.examples.append(example)
            if len(result.examples) >= options.target_count:
                break
        result.batches = batch_index
        if on_batch:
            on_batch(
                batch_index,
                {
                    "ok": True,
                    "generated_from_batch": batch_generated,
                    "accepted_total": len(result.examples),
                    "target": options.target_count,
                },
            )
        if len(result.examples) >= options.target_count:
            break
        if not raw_examples and batch_index >= 5 and not result.examples:
            # Model produced nothing useful in 5 consecutive empty batches.
            result.failed_batches.append("Model bos dizi dondurmeye devam etti; durduruluyor.")
            break

    if result.failed_batches and not result.examples and options.target_count > 0:
        print(
            "Uyari: sentetik ornek uretimi tamamen basarisiz; aktif ornek sayisi 0.",
            file=sys.stderr,
        )
    return result
