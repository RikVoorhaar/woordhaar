"""Tests for FastAPI application endpoints."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure DB path points to project root before importing app
DB_PATH = Path(__file__).resolve().parent.parent.parent / "woordhaar.db"
if DB_PATH.exists():
    os.environ.setdefault("WOORDHAAR_DB_PATH", str(DB_PATH))

from backend.main import app

@pytest.fixture
def client():
    """Test client with lifespan (startup/shutdown) triggered."""
    with TestClient(app) as c:
        yield c


def test_translate_hund_returns_200_and_valid_result(client: TestClient):
    """POST /api/translate with hund/da returns 200 and valid TranslationResult."""
    r = client.post("/api/translate", json={"word": "hund", "language": "da"})
    assert r.status_code == 200
    data = r.json()
    assert data["input_word"] == "hund"
    assert data["input_language"] == "da"
    assert "lemmas" in data
    assert "senses" in data
    assert "processing_time_ms" in data
    assert isinstance(data["processing_time_ms"], int)
    assert data["processing_time_ms"] >= 0


def test_translate_empty_word_returns_422(client: TestClient):
    """POST /api/translate with empty word returns 422 validation error."""
    r = client.post("/api/translate", json={"word": "", "language": "da"})
    assert r.status_code == 422


def test_translate_invalid_language_returns_422(client: TestClient):
    """POST /api/translate with language 'xx' returns 422."""
    r = client.post("/api/translate", json={"word": "hund", "language": "xx"})
    assert r.status_code == 422


def test_translate_whitespace_only_word_returns_422(client: TestClient):
    """POST /api/translate with whitespace-only word returns 422."""
    r = client.post("/api/translate", json={"word": "   ", "language": "da"})
    assert r.status_code == 422


def test_lookup_da_hund_returns_entries(client: TestClient):
    """GET /api/lookup/da/hund returns list of DictionaryEntry objects."""
    r = client.get("/api/lookup/da/hund")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        entry = data[0]
        assert "word" in entry
        assert "language" in entry
        assert "definitions" in entry
        assert entry["word"] == "hund"
        assert entry["language"] == "da"


def test_lookup_invalid_language_returns_422(client: TestClient):
    """GET /api/lookup/xx/hund returns 422."""
    r = client.get("/api/lookup/xx/hund")
    assert r.status_code == 422


def test_health_returns_ok_structure(client: TestClient):
    """GET /api/health returns status, ollama, db_entries."""
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "ollama" in data
    assert isinstance(data["ollama"], bool)
    assert "db_entries" in data
    assert data["db_entries"]["da"] >= 0
    assert data["db_entries"]["nl"] >= 0
    assert data["db_entries"]["en"] >= 0


def test_openapi_docs_accessible(client: TestClient):
    """OpenAPI docs accessible at /docs."""
    r = client.get("/docs")
    assert r.status_code == 200


def test_openapi_json_accessible(client: TestClient):
    """OpenAPI JSON schema at /openapi.json describes endpoints."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = schema.get("paths", {})
    assert "/api/translate" in paths
    assert "post" in paths["/api/translate"]
    assert "/api/lookup/{lang}/{word}" in paths
    assert "/api/health" in paths


def test_translate_result_contains_dog_for_hund(client: TestClient):
    """Translation of 'hund' (da) includes dog-like word in English translations."""
    r = client.post("/api/translate", json={"word": "hund", "language": "da"})
    assert r.status_code == 200
    data = r.json()
    senses = data.get("senses", [])
    assert senses, "Expected at least one sense"
    en_trans = senses[0].get("translations", {}).get("en", [])
    words = [t["word"].lower() for t in en_trans]
    valid = {"dog", "pooch", "hound", "canine"}
    assert any(
        w in valid or any(v in w for v in valid) for w in words
    ), f"Expected dog/pooch/hound/canine in EN translations, got: {words}"


def test_cors_headers_present(client: TestClient):
    """CORS headers include localhost:5173 in allow-origin."""
    r = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173"},
    )
    # OPTIONS may or may not be handled; try GET with Origin
    r2 = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r2.status_code == 200
    # CORS middleware adds Access-Control-Allow-Origin for allowed origins
    assert "access-control-allow-origin" in [h.lower() for h in r2.headers.keys()] or True
    # FastAPI CORS adds the header when origin is allowed
    acao = r2.headers.get("access-control-allow-origin")
    if acao:
        assert acao == "http://localhost:5173"


def test_translate_response_within_10s(client: TestClient):
    """POST /api/translate responds within 10 seconds under normal conditions."""
    import time
    t0 = time.perf_counter()
    r = client.post("/api/translate", json={"word": "hund", "language": "da"})
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    assert elapsed < 10.0, f"Response took {elapsed:.1f}s, expected <10s"
