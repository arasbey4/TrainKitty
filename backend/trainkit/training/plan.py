"""Training plan generation + validation.

The plan is machine-validated against the real environment (deps, dataset,
base model fields). AI can suggest values, but every plan is re-validated here
before any training process starts.
"""

from __future__ import annotations

from typing import Any, Literal

import pydantic

AVAILABLE_MODES = ("configuration_only", "lora", "full_finetune")


class TrainingConfig(pydantic.BaseModel):
    """Validated training configuration (persisted verbatim for each run)."""

    mode: Literal["configuration_only", "lora", "full_finetune"] = "lora"
    base_model_id: str
    base_model_revision: str = "main"
    dataset_id: str
    dataset_revision: str | None = None
    dataset_path: str
    validation_dataset_path: str | None = None
    tokenizer_source: str | None = None
    chat_template: str | None = None
    sequence_length: int = pydantic.Field(default=512, ge=32, le=8192)
    per_device_batch_size: int = pydantic.Field(default=1, ge=1, le=256)
    gradient_accumulation_steps: int = pydantic.Field(default=4, ge=1, le=256)
    learning_rate: float = pydantic.Field(default=2e-4, gt=0, le=1)
    num_epochs: int = pydantic.Field(default=3, ge=1, le=100)
    warmup_ratio: float = pydantic.Field(default=0.03, ge=0, le=1)
    weight_decay: float = pydantic.Field(default=0.0, ge=0, le=1)
    precision: Literal["fp32", "fp16", "bf16", "auto"] = "auto"
    gradient_checkpointing: bool = False
    lora_rank: int = pydantic.Field(default=8, ge=1, le=512)
    lora_alpha: int = pydantic.Field(default=16, ge=1, le=1024)
    lora_dropout: float = pydantic.Field(default=0.05, ge=0, le=1)
    lora_target_modules: list[str] | None = None
    eval_interval: int = pydantic.Field(default=50, ge=1)
    save_steps: int = pydantic.Field(default=100, ge=1)
    seed: int = pydantic.Field(default=42, ge=0)
    output_dir: str | None = None
    hardware: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    def to_run_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PlanIssue(pydantic.BaseModel):
    severity: Literal["error", "warning", "info"]
    field: str
    message: str


def validate_plan(config: TrainingConfig, dataset_schema: str | None = None) -> list[PlanIssue]:
    issues: list[PlanIssue] = []
    if config.precision == "fp16" and config.hardware not in ("auto", "cuda"):
        issues.append(
            PlanIssue(severity="warning", field="precision",
                      message="fp16 CPU/MPS üzerinde kararsız olabilir; fp32 veya bf16 önerilir.")
        )
    if config.mode == "lora" and config.lora_target_modules:
        issues.append(
            PlanIssue(severity="info", field="lora_target_modules",
                      message="Hedef modüller model mimarisine göre doğrulanacaktır.")
        )
    if dataset_schema == "chat" and config.chat_template is None and not config.dataset_path:
        issues.append(
            PlanIssue(severity="warning", field="chat_template",
                      message="Sohbet biçimli veri için chat şablonu belirtilmedi.")
        )
    est_tokens_per_step = (
        config.sequence_length * config.per_device_batch_size * config.gradient_accumulation_steps
    )
    issues.append(
        PlanIssue(severity="info", field="compute",
                  message=f"Adım başına yaklaşık token girdisi: {est_tokens_per_step} (tahmin)")
    )
    return issues


def default_plan_for_dataset(
    base_model_id: str,
    dataset_id: str,
    dataset_path: str,
    validation_dataset_path: str | None,
    dataset_schema: str | None,
    hardware_summary: dict[str, Any],
) -> TrainingConfig:
    """Sensible defaults shaped by measured hardware (recommendations only)."""
    cuda = hardware_summary.get("gpu", {}).get("cuda", {}).get("available")
    total_ram = (hardware_summary.get("ram") or {}).get("total_bytes") or 0
    seq_len = 512
    batch = 2 if cuda else 1
    grad_accum = 8 if not cuda else 4
    precision = "auto"
    grad_ckpt = True
    if not cuda and total_ram < 4 * 1024**3:
        seq_len = 256
    return TrainingConfig(
        mode="lora",
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        validation_dataset_path=validation_dataset_path,
        dataset_schema=dataset_schema,
        sequence_length=seq_len,
        per_device_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        precision=precision,  # type: ignore[arg-type]
        gradient_checkpointing=grad_ckpt,
        learning_rate=2e-4,
        num_epochs=3,
        eval_interval=50,
        save_steps=100,
        seed=42,
        hardware="auto",
    )
