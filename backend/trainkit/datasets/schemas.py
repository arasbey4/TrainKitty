"""Dataset example schemas, detection and normalization.

Supported shapes (per example):
- instruction-tuning: {"instruction", "input", "output"}
- chat: {"messages": [{role, content}, ...]}
- prompt/completion: {"prompt", "response"} or (legacy) {"prompt", "completion"}
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

VALID_SCHEMA_TYPES = ("instruction", "chat")


class ChatMessage(BaseModel):
    role: str
    content: str


class InstructionExample(BaseModel):
    instruction: str
    input: str = ""
    output: str


class ChatExample(BaseModel):
    messages: list[ChatMessage]


def normalize_role(role: str) -> str:
    r = (role or "user").strip().lower()
    aliases = {"assistant": "assistant", "bot": "assistant", "model": "assistant", "gpt": "assistant", "system": "system", "human": "user", "user": "user"}
    return aliases.get(r, r)


def detect_example_kind(raw: dict[str, Any]) -> str | None:
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("messages"), list) and raw["messages"]:
        return "chat"
    if "instruction" in raw or ("prompt" in raw and ("response" in raw or "completion" in raw or "output" in raw)):
        return "instruction"
    return None


def normalize_example(raw: Any, *, index: int, source_id: str | None = None) -> tuple[str, dict[str, Any]] | None:
    """Return (kind, normalized_example) or None when the record is unusable."""
    if not isinstance(raw, dict):
        return None
    kind = detect_example_kind(raw)
    if kind == "chat":
        msgs = raw["messages"]
        normalized_msgs: list[dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                return None
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                return None
            normalized_msgs.append({"role": normalize_role(m.get("role", "")), "content": content})
        if len(normalized_msgs) < 2:
            return None
        example = {"messages": normalized_msgs}
        _stamp(example, index, source_id, raw)
        return "chat", example
    if kind == "instruction":
        instruction = raw.get("instruction") or raw.get("prompt")
        output = raw.get("output", raw.get("response", raw.get("completion")))
        if not isinstance(instruction, str) or not instruction.strip():
            return None
        if not isinstance(output, str) or not output.strip():
            return None
        inp = raw.get("input", "")
        if not isinstance(inp, str):
            inp = "" if inp is None else str(inp)
        example = {
            "instruction": instruction.strip(),
            "input": inp.strip(),
            "output": output.strip(),
        }
        _stamp(example, index, source_id, raw)
        return "instruction", example
    return None


def _stamp(example: dict[str, Any], index: int, source_id: str | None, raw: dict[str, Any]) -> None:
    """Attach deterministic identity + provenance without touching content."""
    import hashlib
    import json

    content = json.dumps(example, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    example["_id"] = f"ex_{digest}_{index:06d}"
    example["_source"] = source_id
    example["_index"] = index


def examples_hash(example: dict[str, Any]) -> str:
    import hashlib
    import json

    payload = {k: example.get(k) for k in ("instruction", "input", "output", "messages") if k in example}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def iter_jsonl(path) -> Iterator[dict[str, Any]]:
    import json

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"JSONL ayrıştırma hatası: {exc}") from exc
