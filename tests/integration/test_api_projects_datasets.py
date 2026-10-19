"""API integration tests (real FastAPI app, real temp SQLite DB)."""

import json

from fastapi.testclient import TestClient

from trainkit.api.app import create_app
from trainkit.database import db_file_path


import pytest


@pytest.fixture()
def client(tmp_data_dir):
    app = create_app()
    return TestClient(app)


def test_health_and_docs(client):
    c = client
    h = c.get("/api/health").json()
    assert h["status"] == "ok"
    assert h.get("app") == "trainkit"


def test_project_crud_and_persistence(client):
    created = client.post("/api/projects", json={"name": "Atlas", "description": "test"}).json()
    pid = created["id"]

    r = client.patch(
        f"/api/projects/{pid}",
        json={
            "spec": {"target_model_name": "Atlas", "purpose": "Python asistanı"},
            "spec_approved": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["spec"]["target_model_name"] == "Atlas"

    # fresh client reuses same DB file, confirms persistence
    r2 = client.get(f"/api/projects/{pid}")
    assert r2.status_code == 200
    assert r2.json()["spec_approved"] is True
    assert r2.json()["spec"]["purpose"] == "Python asistanı"

    assert client.delete(f"/api/projects/{pid}").status_code == 428
    assert client.delete(f"/api/projects/{pid}?confirm=true").status_code == 200
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_dataset_upload_and_prepare(client):
    rows = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(5)]
    rows.append({"instruction": "Q0", "output": "A0"})  # exact dupe
    rows.append({"malformed": True})
    content = "\n".join(json.dumps(r) for r in rows).encode()

    up = client.post(
        "/api/datasets/upload",
        files={"file": ("data.jsonl", content, "application/octet-stream")},
    )
    assert up.status_code == 200
    ds = up.json()
    assert ds["example_count"] == 7
    assert ds["source"] == "upload"

    prep = client.post(
        "/api/datasets/prepare",
        json={"dataset_id": ds["id"], "validation_split": 0.2, "seed": 1},
    )
    assert prep.status_code == 200
    body = prep.json()
    assert body["prepared"]["status"] == "validated"
    report = body["report"]["report"]
    assert report["duplicates"] == 1
    assert report["rejected"] == 1  # malformed
    assert report["train_count"] == 4
    assert report["validation_count"] == 1


def test_restore_upload_rejects_bad_extension(client):
    r = client.post(
        "/api/datasets/upload",
        files={"file": ("virus.exe", b"MZ", "application/octet-stream")},
    )
    assert r.status_code == 400
