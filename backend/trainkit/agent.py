"""Bounded agent orchestration on top of the Ollama client.

The local model proposes/interprets; trusted application code executes typed
tools bound to real services. No tool returns hardcoded success; no raw model
text is ever executed as a shell command; no approval bypass exists.
"""

from __future__ import annotations

import json
from typing import Any

import pydantic

from .db_models import AgentRun, new_id
from .ollama.client import OllamaClient
from .projects import ProjectSpec

SUPPORTED_TOOLS = {
    "list_ollama_models",
    "inspect_hardware",
    "inspect_dataset",
    "discover_huggingface_dataset",
    "validate_dataset",
    "create_training_plan",
    "run_resource_preflight",
}
STATEMENT_TOOL_LIST = ", ".join(sorted(SUPPORTED_TOOLS))


class AgentConfigError(Exception):
    pass


class SpecDraft(pydantic.BaseModel):
    """Structured project specification proposed by the local model."""

    target_model_name: str | None = None
    purpose: str | None = None
    intended_users: str | None = None
    languages: list[str] = pydantic.Field(default_factory=list)
    personality: str | None = None
    system_instructions: str | None = None
    desired_behavior: list[str] = pydantic.Field(default_factory=list)
    unwanted_behavior: list[str] = pydantic.Field(default_factory=list)
    example_inputs: list[str] = pydantic.Field(default_factory=list)
    output_style: str | None = None
    dataset_requirements: str | None = None
    evaluation_criteria: list[str] = pydantic.Field(default_factory=list)
    base_model_preference: str | None = None
    training_method: str | None = None
    resource_constraints: str | None = None
    export_requirements: str | None = None
    missing_information: list[str] = pydantic.Field(default_factory=list)


SPEC_PROMPT = """Bir kullanıcı, yerel cihazında eğitilecek özel bir model istiyor.
Kullanıcı girdisi:
---
{user_description}
---

Bu tanımı aşağıdaki şemaya dökülmüş bir spec'e dönüştür ve YALNIZCA şu alanları
içeren bir JSON nesnesi döndür: target_model_name, purpose, intended_users,
languages, personality, system_instructions, desired_behavior,
unwanted_behavior, example_inputs, output_style, dataset_requirements,
evaluation_criteria, base_model_preference, training_method,
resource_constraints, export_requirements, missing_information.
Kritik bilgi eksikse missing_information dizisine kısa sorularla ekle. Yalnızca
talimat verilemediğini varsayma; varsayımları missing_information'a not et."""


def draft_spec(
    db,
    client: OllamaClient,
    model: str,
    user_description: str,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Parse the natural-language description into a structured spec."""
    agent_run = AgentRun(id=new_id("agent"), project_id=project_id, kind="spec_generation")
    db.add(agent_run)
    db.commit()
    agent_run.input = {"description": user_description, "model": model}
    try:
        if not user_description or not user_description.strip():
            raise AgentConfigError("Boş açıklama; spec üretilemedi.")
        data = client.generate_json(
            model,
            SPEC_PROMPT.format(user_description=user_description),
            schema_hint="target_model_name/purpose/languages/system_instructions/.../missing_information",
            system=(
                "Ürün: sistem talimatı üreten bir model yaratma motorusun. Görevin, "
                "kullanıcı tanımını belirtilen alanlarla bir JSON spec'e dönüştürmek."
            ),
        )
        draft = SpecDraft.model_validate(data)
        agent_run.status = "completed"
        agent_run.output = draft.model_dump()
        db.commit()
        return draft.model_dump()
    except Exception as exc:
        user_message = getattr(exc, "user_message", None) or str(exc)
        agent_run.status = "failed"
        agent_run.error = user_message[:2000]
        db.commit()
        raise


class _FailedToolError(Exception):
    """Tool result surface; the model cannot override tool outcomes."""


def safe_tool_call(agent_result: dict[str, Any], expected_tool: str) -> Any:
    """Type-check an agent tool call against a known tool name.

    The agent can only *request* whitelisted tool names; payload contents are
    validated by each tool's own API-layer schema.
    """
    tool = agent_result.get("tool")
    if tool not in SUPPORTED_TOOLS:
        _err = (
            f"Aracı tanımlanmadı: {tool}. İzinli araçlar: {STATEMENT_TOOL_LIST}"
        )
        raise _FailedToolError(_err)
    if tool != expected_tool:
        raise _FailedToolError(f"Beklenen araç: {expected_tool}; gelen: {tool}")
    return agent_result.get("args") or {}


def interpret_error(client: OllamaClient, model: str, error_message: str) -> dict[str, Any]:
    """Ask the local model for a human-friendly, bounded interpretation of an error."""
    prompt = (
        "Aşağıdaki hata mesajını basit Türkçe ile özetle ve yapıcı bir öneri ekle. "
        "Yalnızca şu yapıda JSON döndür: "
        '{"summary": "...", "suggestion": "..."}. Neden: ' + error_message[:1000]
    )
    try:
        data = client.generate_json(model, prompt, schema_hint='{"summary","suggestion"}')
        summary = str(data.get("summary", ""))[:500]
        suggestion = str(data.get("suggestion", ""))[:500]
        return {"summary": summary, "suggestion": suggestion, "source": "ollama_heuristic"}
    except Exception:
        return {
            "summary": None,
            "suggestion": None,
            "source": "unavailable",
        }
