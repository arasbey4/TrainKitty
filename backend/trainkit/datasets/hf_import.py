"""Hugging Face dataset import via official APIs.

Metadata is fetched from the public HF API; loading actual records is done
with the `datasets` library when installed. Gated/private access is respected;
nothing here bypasses access controls.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..security import validate_safe_hf_repo_id

HF_API = "https://huggingface.co/api"


class HFDatasetError(Exception):
    def __init__(self, user_message: str, detail: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


def discover_dataset(repo_id: str) -> dict[str, Any]:
    """Fetch real dataset metadata (features, size estimate, license)."""
    repo_id = validate_safe_hf_repo_id(repo_id)
    try:
        resp = httpx.get(f"{HF_API}/datasets/{repo_id}?full=true", timeout=30.0)
    except httpx.ConnectError as exc:
        raise HFDatasetError(
            "Hugging Face'e bağlanılamadı (internet bağlantısı gerekli).", str(exc)
        ) from exc
    except httpx.TimeoutException as exc:
        raise HFDatasetError("Hugging Face isteği zaman aşımına uğradı.", str(exc)) from exc
    if resp.status_code == 404:
        raise HFDatasetError(f"Veri kümesi bulunamadı: {repo_id}")
    if resp.status_code in (401, 403):
        raise HFDatasetError(
            f"{repo_id} gizli veya erişim izni gerektiren bir veri kümesi; "
            "yetkili kimlik bilgisi gerekli. Erişim kontrolü aşılamaz."
        )
    resp.raise_for_status()
    data = resp.json()

    tags = data.get("tags") or []
    license_name = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    card = data.get("cardData") or {}
    dataset_info = card.get("dataset_info") if isinstance(card.get("dataset_info"), dict) else {}
    features = data.get("features") or dataset_info.get("features")
    size = data.get("size") if isinstance(data.get("size"), dict) else {}
    siblings = data.get("siblings") or []
    file_names = [s.get("rfilename") for s in siblings[:50] if s.get("rfilename")]

    return {
        "id": data.get("id"),
        "private": bool(data.get("private")),
        "gated": bool(data.get("gated")) or "gated" in tags,
        "features": features,
        "size_bytes_estimate": size.get("dataset"),
        "num_files_parquet": size.get("num_files_parquet"),
        "download_count": data.get("downloads"),
        "license": license_name,
        "files": file_names,
        "tags": tags[:30],
        "provenance_note": "Boyut/lisans bilgisi Hugging Face API'sinden alındı; boyut tahminidir.",
        "references_note": "meta_huggingface_api",
    }


def load_dataset_records(
    repo_id: str,
    *,
    split: str = "train",
    config_name: str | None = None,
    max_records: int = 1000,
    token: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load real records with the `datasets` library (official mechanism)."""
    from ..hardware import _module_available

    repo_id = validate_safe_hf_repo_id(repo_id)
    if not _module_available("datasets"):
        raise HFDatasetError(
            "Hugging Face yükleme için `datasets` kütüphanesi kurulu değil. Kurulum: "
            "pip install 'trainkit[hf]' (veya pip install datasets)."
        )
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HFDatasetError("datasets kütüphanesi içe aktarılamadı.", str(exc)) from exc
    try:
        kwargs: dict[str, Any] = {"split": split, "token": token}
        if config_name:
            kwargs["name"] = config_name
        ds = hf_load(repo_id, **kwargs)
    except Exception as exc:
        raise HFDatasetError(
            f"Veri kümesi yüklenemedi: {repo_id} (split={split}, config={config_name}). "
            f"Ayrıntı: {str(exc)[:400]}"
        ) from exc

    features = {k: str(getattr(f, "_type", v)) for k, v in getattr(ds, "features", {}).items()}
    total = int(len(ds))
    records: list[dict[str, Any]] = []
    for i in range(int(min(max_records, total))):
        rec = _clean_record(ds[i])
        if rec is not None:
            records.append(rec)
    info = {
        "repo_id": repo_id,
        "split": split,
        "config_name": config_name,
        "total_records": total,
        "loaded_records": len(records),
        "features": features,
        "limit_applied": max_records if total > max_records else None,
    }
    return records, info


def _clean_record(row: dict[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, (int, float, bool)) and value is not None and not isinstance(value, bool):
            out[key] = str(value)
        elif isinstance(value, (int, bool)) or value is None:
            out[key] = value
        elif isinstance(value, float):
            out[key] = str(value)
        elif isinstance(value, dict):
            out[key] = {
                str(k): v if isinstance(v, str) else str(v)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            out[key] = [str(x) for x in value]
        else:
            out[key] = str(value)
    return out


def map_records_to_schema(
    records: list[dict[str, Any]], mapping: dict[str, str | None]
) -> list[dict[str, Any]]:
    """Map user-selected columns onto the instruction schema."""
    out: list[dict[str, Any]] = []
    for rec in records:

        def pick(column: str | None) -> str:
            if not column:
                return ""
            value = rec.get(column)
            if isinstance(value, str):
                return value.strip()
            if value is None:
                return ""
            return str(value).strip()

        out.append(
            {
                "instruction": pick(mapping.get("instruction")),
                "input": pick(mapping.get("input")),
                "output": pick(mapping.get("output")),
            }
        )
    return out


def license_check(meta: dict[str, Any]) -> dict[str, Any]:
    lic = meta.get("license")
    if not lic:
        return {
            "status": "missing",
            "warning": "Lisans bilgisi bulunamadı; kullanım kısıtlarını kullanmadan önce gözden geçirin.",
        }
    if str(lic).lower() in ("other", "unknown"):
        return {
            "status": "review",
            "warning": "Lisans 'other' olarak işaretlenmiş; özel şartları kullanmadan önce inceleyin.",
        }
    return {"status": "known", "license": lic, "warning": None}
