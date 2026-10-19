"""Real Ollama HTTP client.

Talks to the locally running Ollama instance using its documented HTTP API:
- GET  /api/tags        -> installed models
- POST /api/generate    -> completion (supports format:"json" structured output)
- POST /api/chat        -> chat
- POST /api/pull        -> model pull (streaming progress)
- POST /api/create      -> create a model from a Modelfile

No external/cloud fallback exists anywhere in this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import httpx

from ..settings import get_setting
from . import (
    OllamaConnectionError,
    OllamaContextTooLongError,
    OllamaInvalidOutputError,
    OllamaModelNotFoundError,
    OllamaRequestCancelled,
    OllamaResponseError,
    OllamaTimeoutError,
)

REQUEST_CANCELED_HINTS = ("cancel", "abort")


class OllamaClient:
    """A thin, real HTTP client for a locally running Ollama server.

    A client instance is cheap; construct one per operation with the current
    settings so endpoint changes take effect immediately.
    """

    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s

    @classmethod
    def from_settings(cls, db) -> "OllamaClient":
        return cls(
            base_url=get_setting(db, "ollama.base_url"),
            timeout_s=get_setting(db, "ollama.request_timeout_s"),
        )

    @classmethod
    def direct(cls, base_url: str, timeout_s: float = 120.0) -> "OllamaClient":
        return cls(base_url=base_url, timeout_s=timeout_s)

    # ------------------------------------------------------------------ core

    def _post(self, path: str, payload: dict[str, Any], *, timeout_s: float | None = None):
        base = self.base_url or ""
        if not base:
            raise OllamaConnectionError("(endpoint ayarlanmamış)")
        try:
            return httpx.post(
                base.rstrip("/") + path,
                json=payload,
                timeout=timeout_s or self.timeout_s or 120.0,
            )
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(base, str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout_s or 120.0, str(exc)) from exc

    def status(self) -> dict[str, Any]:
        server_base = self.base_url
        try:
            resp = httpx.get(
                (server_base or "").rstrip("/") + "/api/tags",
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError as exc:
            return {"connected": False, "base_url": server_base, "error": str(exc)}
        except Exception as exc:  # httpx.HTTPError or JSON problems
            return {"connected": False, "base_url": server_base, "error": str(exc)}
        models = self._parse_models(data)
        return {"connected": True, "base_url": server_base, "models": models}

    # ---------------------------------------------------------------- models

    @staticmethod
    def _parse_models(data: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in data.get("models", []):
            entry: dict[str, Any] = {"name": m.get("name") or m.get("model")}
            details = m.get("details") or {}
            size_bytes = m.get("size")
            entry["size_bytes"] = size_bytes if size_bytes is not None else None
            # Real metadata only: never invent parameter counts or families.
            entry["parameter_size"] = details.get("parameter_size") or None
            entry["family"] = details.get("family") or None
            entry["quantization_level"] = details.get("quantization_level") or None
            entry["digest"] = m.get("digest")
            entry["modified_at"] = m.get("modified_at")
            out.append(entry)
        return out

    def list_models(self) -> list[dict[str, Any]]:
        base = self.base_url
        try:
            resp = httpx.get(
                (base or "").rstrip("/") + "/api/tags", timeout=self.timeout_s or 30.0
            )
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(base) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(self.timeout_s or 30.0) from exc
        try:
            resp.raise_for_status()
            data = resp.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama geçersiz yanıt döndürdü (JSON çözümlenemedi).") from exc
        return self._parse_models(data)

    def model_exists(self, model: str) -> bool:
        try:
            models = self.list_models()
        except OllamaConnectionError:
            raise
        names = {m.get("name") or m.get("model") for m in models}
        return model in names or model.split(":")[0] in {n.split(":")[0] for n in names if n}

    # ------------------------------------------------------- generation core

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        format: str | None = None,  # "json" enables Ollama's structured mode
        options: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        cancel_token: dict[str, bool] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        resp = self._post("/api/generate", payload, timeout_s=timeout_s)
        return self._handle_generate_response(resp, cancel_token=cancel_token)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        format: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options
        resp = self._post("/api/chat", payload, timeout_s=timeout_s)
        self._check_response(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama geçersiz chat yanıtı döndürdü.") from exc
        return data

    @staticmethod
    def _check_response(resp: httpx.Response) -> None:
        if resp.status_code == 404:
            try:
                detail = resp.json().get("error", "")
            except ValueError:
                detail = ""
            raise OllamaModelNotFoundError(model=detail or "unknown", detail=resp.text[:2000])
        if resp.status_code >= 400:
            try:
                err = resp.json()
                err_text = err.get("error") or str(err)
            except ValueError:
                err_text = resp.text[:2000]
            lowered = err_text.lower()
            if "context" in lowered and ("length" in lowered or "exceed" in lowered):
                raise OllamaContextTooLongError(err_text)
            raise OllamaResponseError(f"Ollama hatası: {err_text}", detail=f"HTTP {resp.status_code}")

    def _handle_generate_response(self, resp: httpx.Response, *, cancel_token=None) -> str:
        if cancel_token and cancel_token.get("cancelled"):
            raise OllamaRequestCancelled()
        self._check_response(resp)
        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama geçersiz generate yanıtı döndürdü.") from exc
        if data.get("done") is False and not data.get("response"):
            raise OllamaResponseError("Ollama yanıtı beklenenden erken kesildi.")
        text = data.get("response")
        if text is None:
            raise OllamaResponseError("Ollama yanıtında 'response' alanı bulunamadı.")
        return text

    # --------------------------------------------------- structured JSON flow

    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull a JSON document out of model output (fenced or bare)."""
        text = text.strip()
        fence = OllamaClient._JSON_FENCE_RE.search(text)
        if fence:
            return fence.group(1).strip()
        # Otherwise find the outermost {...} or [...]
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return text

    def generate_json(
        self,
        model: str,
        prompt: str,
        schema_hint: str,
        *,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        max_repair_attempts: int = 2,
        cancel_token: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Structured generation with bounded repair.

        1. Ask Ollama for JSON (format:"json" where supported).
        2. Parse + validate via caller-supplied pydantic model by the caller
           returning ``schema_hint`` describing expected shape. Validation is
           performed by the caller; here we only guarantee parseable JSON.
        """
        options = dict(options or {})
        options.setdefault("temperature", 0.4)
        system = system or (
            "Sen bir veri motif motorusmusak değil. Yalnızca istenen JSON belgesini "
            "döndür. Açıklama, giriş veya başka metin ekleme."
        )
        attempt_text = self.generate(
            model,
            prompt,
            system=system,
            format="json",
            options=options,
            cancel_token=cancel_token,
        )
        failures: list[str] = []
        for attempt in range(max_repair_attempts + 1):
            raw = self._extract_json(attempt_text)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("JSON bir nesne ({...}) olmalı")
            except ValueError as exc:
                failures.append(f"attempt {attempt + 1}: {exc}")
                if attempt == max_repair_attempts:
                    break
                repair_prompt = (
                    f"Önceki yanıtın geçerli JSON'a uyması için düzeltme gerekiyor.\n\n"
                    f"Beklenen yapı:\n{schema_hint}\n\n"
                    f"Önceki yanıt:\n{attempt_text[:4000]}\n\n"
                    f"Düzeltmeleri uygula ve SADECE geçerli JSON döndür."
                )
                attempt_text = self.generate(
                    model,
                    repair_prompt,
                    system=system,
                    format="json",
                    options={**options, "temperature": 0.0},
                    cancel_token=cancel_token,
                )
        raise OllamaInvalidOutputError(
            "Model çıktısı gerekli JSON yapısına dönüştürülemedi.",
            detail=" | ".join(failures)[:2000],
        )

    def generate_text(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        return self.generate(
            model, prompt, system=system, options=options, timeout_s=timeout_s
        )

    # ------------------------------------------------------------------ pull

    def pull(self, model: str, cancel: Any = None):
        """Stream a model pull. Yields dict progress lines from the real API.

        Streams must be consumed by the caller (e.g. as an SSE endpoint).
        """
        base = self.base_url
        payload = {"name": model, "stream": True}
        try:
            with httpx.stream(
                "POST",
                (base or "").rstrip("/") + "/api/pull",
                json=payload,
                timeout=httpx.Timeout(self.timeout_s or 120.0, read=None),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        event = {"status": line}
                    yield event
                    if event.get("error"):
                        raise OllamaResponseError(
                            f"Model indirme başarısız: {event['error']}", detail=json.dumps(event)
                        )
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(base) from exc

    # -------------------------------------------------- model creation (import)

    def create_model_stream(self, model_name: str, modelfile: str):
        """Create an Ollama model from a Modelfile; yields streaming events."""
        base = self.base_url
        payload = {"model": model_name, "modelfile": modelfile, "stream": True}
        try:
            with httpx.stream(
                "POST",
                (base or "").rstrip("/") + "/api/create",
                json=payload,
                timeout=httpx.Timeout(self.timeout_s or 120.0, read=None),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        event = {"status": line}
                    if event.get("error"):
                        yield event
                        return
                    yield event
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(base) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else ""
            raise OllamaResponseError(
                f"Ollama modeli oluşturma başarısız (HTTP {exc.response.status_code}).", detail
            ) from exc
