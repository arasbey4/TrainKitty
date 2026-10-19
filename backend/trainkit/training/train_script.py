"""Real fine-tuning script (runs as its own process).

Usage: python -m trainkit.training.train_script <config.json>

Reads a run config JSON, trains real adapter/weights with PyTorch +
Transformers + PEFT, and emits JSON lines on stdout:

  {"type": "state",   "state": "..."}
  {"type": "log",     "message": "..."}
  {"type": "metric",  "step": N, "loss": x, "lr": y, "epoch": z}
  {"type": "error",   "error": "..."}
  {"type": "result",  ...}            # final summary; file also written

Nothing is simulated: every metric comes from the actual training loop; if
the process dies, no 'completed' state is ever emitted.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

EMIT_LINE_MAX = 500


def emit(payload: dict[str, Any]) -> None:
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def log(message: str) -> None:
    emit({"type": "log", "message": message[:EMIT_LINE_MAX]})


def fail(message: str, error_class: str = "TrainingError") -> None:
    emit({"type": "error", "error": message[:EMIT_LINE_MAX], "error_class": error_class})
    sys.exit(1)


def instruction_text(example: dict[str, Any]) -> str:
    prompt = f"### Görev:\n{example.get('instruction', '')}\n"
    if example.get("input"):
        prompt += f"\n### Girdi:\n{example['input']}\n"
    prompt += f"\n### Yanıt:\n{example.get('output', '')}"
    return prompt


def chat_text(example: dict[str, Any]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in example.get("messages", []))


def load_examples(path: str, max_records: int = 200_000) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if len(examples) >= max_records:
                break
    return examples


def detect_device(preference: str) -> str:
    import torch

    if preference == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(requested: str, device: str) -> str:
    if requested and requested != "auto":
        return requested
    if device == "cuda":
        import torch

        try:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return "bf16"
        except Exception:
            pass
        return "fp16"
    return "fp32"


def main() -> None:
    if len(sys.argv) < 2:
        fail("config.json yolu eksik", "ConfigError")
    config_path = Path(sys.argv[1])
    if not config_path.is_file():
        fail(f"config bulunamadı: {config_path}", "ConfigError")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        fail(f"config JSON hatası: {exc}", "ConfigError")

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- dependency check (real imports; fail honestly) ----------------------
    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        fail(f"Eğitim bağımlılıkları yüklenemedi: {exc}", "DependencyError")

    seed = int(config.get("seed", 42))
    import torch
    from transformers import set_seed

    set_seed(seed)

    mode = config.get("mode", "lora")
    device = detect_device(config.get("hardware", "auto"))
    precision = resolve_precision(config.get("precision", "auto"), device)
    log(f"Mod: {mode} | cihaz: {device} | hassasiyet: {precision} | torch {torch.__version__}")

    base_model_id = config["base_model_id"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    emit({"type": "state", "state": "preparing"})

    # --- dataset --------------------------------------------------------------
    train_examples = load_examples(config["dataset_path"])
    if not train_examples:
        fail("Veri kümesi boş", "DatasetError")
    schema = "chat" if train_examples[0].get("messages") else "instruction"
    to_text = chat_text if schema == "chat" else instruction_text

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            revision=config.get("base_model_revision", "main"),
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        fail(f"Tokenizer yüklenemedi ({base_model_id}): {exc}", "HuggingFaceError")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    max_len = int(config.get("sequence_length", 512))

    def encode(example: dict[str, Any]) -> dict[str, list[int]]:
        text = to_text(example)
        full = tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=True)
        ids = full["input_ids"] or [tokenizer.eos_token_id or 0]
        return {"input_ids": ids, "labels": list(ids)}

    from torch.utils.data import Dataset as TorchDataset

    class listDataset(TorchDataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {k: torch.tensor(v, dtype=torch.long) for k, v in encode(self.rows[idx])}

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attention, labels = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad_len = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention.append([1] * len(ids) + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention),
            "labels": torch.stack(labels),
        }

    train_ds = listDataset(train_examples)

    # --- model ---------------------------------------------------------------
    emit({"type": "state", "state": "downloading"})
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(precision, torch.float32 if device == "cpu" else torch.float16)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=config.get("base_model_revision", "main"),
            torch_dtype=torch_dtype,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as exc:
        fail(f"Temel model yüklenemedi ({base_model_id}): {exc}", "HuggingFaceError")
    model.to(device)

    adapter_dir = output_dir / "adapter"
    if mode in ("lora", "full_finetune"):
        emit({"type": "state", "state": "training"})
        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_ds)
                / (
                    int(config.get("per_device_batch_size", 1))
                    * int(config.get("gradient_accumulation_steps", 4))
                )
            ),
        )
        epochs = int(config.get("num_epochs", 3))
        total_steps = steps_per_epoch * epochs
        emit({"type": "state", "state": "validating", "total_steps": total_steps})

        if mode == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as exc:
                fail(f"PEFT yüklenemedi: {exc}", "DependencyError")
            lora_cfg = LoraConfig(
                r=int(config.get("lora_rank", 8)),
                lora_alpha=int(config.get("lora_alpha", 16)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)

        if config.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()

        class EmitCallback(transformers.TrainerCallback):
            def __init__(self) -> None:
                self._lr = None

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                self._lr = logs.get("learning_rate", self._lr)
                payload: dict[str, Any] = {
                    "type": "metric",
                    "step": state.global_step,
                    "total_steps": state.max_steps,
                    "epoch": round(float(state.epoch or 0), 4),
                }
                if "loss" in logs:
                    payload["loss"] = float(logs["loss"])
                if "eval_loss" in logs:
                    payload["eval_loss"] = float(logs["eval_loss"])
                if self._lr is not None:
                    payload["learning_rate"] = float(self._lr)
                emit(payload)

        emit({"type": "state", "state": "training", "total_steps": total_steps})
        eval_ds = None
        validation_path = config.get("validation_dataset_path")
        if validation_path and Path(validation_path).is_file():
            val_examples = load_examples(validation_path, max_records=500)
            if val_examples:
                eval_ds = listDataset(val_examples)

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(config.get("per_device_batch_size", 1)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
            learning_rate=float(config.get("learning_rate", 2e-4)),
            num_train_epochs=epochs,
            warmup_ratio=float(config.get("warmup_ratio", 0.03)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            logging_steps=1,
            save_strategy="steps",
            save_steps=int(config.get("save_steps", 100)),
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=int(config.get("eval_interval", 50)) if eval_ds is not None else None,
            eval_delay=0,
            seed=seed,
            fp16=(precision == "fp16" and device == "cuda"),
            bf16=(precision == "bf16" and device == "cuda"),
            report_to=[],
            dataloader_drop_last=True,
            remove_unused_columns=False,
            label_names=["labels"],
            save_safetensors=True,
            optim="adamw_torch",
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collate,
            callbacks=[EmitCallback()],
        )
        start = time.time()
        try:
            train_result = trainer.train()
        except KeyboardInterrupt:
            fail("Eğitim kullanıcı tarafından iptal edildi", "Cancelled")
        except Exception as exc:
            fail(f"Eğitim süreci hata verdi: {exc}", "TrainingError")
        elapsed = time.time() - start

        # Save adapter (LoRA) or full model.
        emit({"type": "state", "state": "packaging"})
        if mode == "lora":
            model.save_pretrained(str(adapter_dir))
        else:
            model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        metrics = getattr(train_result, "metrics", {}) or {}
        final_loss = metrics.get("train_loss")
        result: dict[str, Any] = {
            "mode": mode,
            "base_model": base_model_id,
            "revision": config.get("base_model_revision", "main"),
            "training_state": "completed",
            "train_examples": len(train_ds),
            "eval_examples": len(eval_ds) if eval_ds is not None else 0,
            "steps_run": getattr(trainer.state, "global_step", None),
            "total_steps_planned": total_steps,
            "train_loss": final_loss,
            "elapsed_s": round(elapsed, 2),
            "device": device,
            "precision": precision,
            "adapter_dir": str(adapter_dir),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        }
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emit({"type": "result", **result})
        emit({"type": "state", "state": "completed"})
        sys.exit(0)

    # ---------------------------------------------------------------- lora end
    # configuration_only: build + tokenize + ONE real base-model inference; no
    # weight training of any kind.
    emit({"type": "state", "state": "preparing"})
    sample = train_examples[0]
    text = to_text(sample)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    model.eval()
    label_text = chat_text(sample) if schema == "chat" else sample.get("output", "")
    with torch.no_grad():
        input_len = ids["input_ids"].shape[1]
        generated_ids = model.generate(
            **ids,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = tokenizer.decode(
        generated_ids[0][input_len:], skip_special_tokens=True
    )
    result = {
        "mode": "configuration_only",
        "training_state": "not_applicable",
        "base_model": base_model_id,
        "sample_prompt": text,
        "sample_reference": label_text,
        "sample_generated": continuation,
        "note": "Bu modda hiçbir model ağırlığı eğitilmedi; yalnızca çıkarım yapılandırması doğrulandı.",
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit({"type": "result", **result})
    emit({"type": "state", "state": "completed"})


if __name__ == "__main__":
    main()
