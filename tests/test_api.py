import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from api import app

# Dynamic API Key fetch matching api.py implementation
api_key = os.environ.get("PAGEINDEX_API_KEY", "remse_default_api_key_789")
headers = {"X-API-Key": api_key}

@pytest.fixture
def client(mocker):
    # Mocking PageIndexREMSE inside api.py
    mocker.patch("api.PageIndexREMSE")
    return TestClient(app)

def test_health_endpoint(client, mocker):
    mocker.patch("llm.ollama_client.check_ollama_connection", return_value=True)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ollama_connected"] is True

def test_unauthorized_endpoints(client):
    # Endpoint requires API Key, so it should fail when key is missing or invalid
    response = client.post("/index-local?pdf_path=dummy.pdf", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403

def test_index_local_endpoint(client, mocker):
    mock_instance = mocker.Mock()
    mock_instance.get_stats.return_value = {"tree_nodes": 5, "total_atoms": 20, "total_triples": 10, "ready": True}
    mocker.patch("api.PageIndexREMSE", return_value=mock_instance)
    mocker.patch("os.path.exists", return_value=True)

    response = client.post(
        "/index-local?pdf_path=dummy.pdf",
        headers=headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["message"] == "Document indexed successfully."
    assert data["stats"]["total_atoms"] == 20

def test_query_endpoint(client, mocker):
    # Register session
    mock_instance = mocker.Mock()
    mock_instance.query.return_value = {
        "answer": "The transformer uses scaled dot-product attention.",
        "selected_sections": ["Section 1"],
        "atoms_used": 5,
        "provenance": {"pages_referenced": [1, 2], "reasoning_path": "Navigated to section 1"}
    }
    
    # Store session in app's internal dict
    import api
    session_id = "test-session"
    api._sessions[session_id] = mock_instance

    response = client.post(
        "/query",
        headers=headers,
        json={"session_id": session_id, "question": "What attention mechanism is used?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The transformer uses scaled dot-product attention."
    assert data["atoms_used"] == 5
    assert "provenance" in data

def test_tutor_chat_endpoint(client, mocker):
    import api
    session_id = "tutor-session"
    mock_instance = mocker.Mock()
    mock_instance.pdf_path = "dummy.pdf"
    mock_instance.doc_id = "dummy-doc"
    mock_instance.query.return_value = {"ordered_atoms": [{"atom_id": 1, "text": "Atom text"}]}
    api._sessions[session_id] = mock_instance

    mocker.patch("api.tutor_engine.chat", return_value="Here is the lesson.")
    mocker.patch("api.tutor_engine.generate_note_card", return_value={"card": "data"})

    response = client.post(
        "/tutor/chat",
        headers=headers,
        json={"session_id": session_id, "message": "Explain self-attention."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Here is the lesson."
    assert data["card_data"] == {"card": "data"}
