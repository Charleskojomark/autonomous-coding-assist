import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import app

client = TestClient(app)

@patch("src.api.main.github_client")
def test_ingest_repo_endpoint_queued(mock_github_client):
    # Mocking GitHub Client calls
    mock_github_client.fetch_repo_files.return_value = []
    
    response = client.post(
        "/ingest",
        json={"repo": "tiangolo/fastapi", "extensions": [".py"]}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "Queued"
    assert response.json()["repo"] == "tiangolo/fastapi"

def test_list_repos_endpoint():
    with patch("src.api.main.vector_store") as mock_vector_store:
        mock_vector_store.list_repos.return_value = ["tiangolo_fastapi", "pallets_flask"]
        
        response = client.get("/repos")
        
        assert response.status_code == 200
        assert "repositories" in response.json()
        assert "tiangolo_fastapi" in response.json()["repositories"]

def test_invalid_repo_name_returns_400():
    response = client.post(
        "/ingest",
        json={"repo": "invalid_name", "extensions": [".py"]}
    )
    
    assert response.status_code == 400
    assert "Invalid repository name" in response.json()["detail"]
