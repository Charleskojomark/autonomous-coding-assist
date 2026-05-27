import os
import pytest
import tempfile
import shutil
from src.models.schemas import CodeChunk
from src.retrieval.vector_store import CodeVectorStore

@pytest.fixture
def temp_vector_store():
    # Use temporary directory for testing persistent ChromaDB and force local embedding
    from unittest.mock import patch
    temp_dir = tempfile.mkdtemp()
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}):
        store = CodeVectorStore(persist_directory=temp_dir, model_name="all-MiniLM-L6-v2")
        yield store
    # Cleanup
    shutil.rmtree(temp_dir)

def test_add_and_search_chunks(temp_vector_store):
    repo = "test/vector-repo"
    
    chunks = [
        CodeChunk(
            file_path="auth.py",
            repo_name=repo,
            content="def verify_jwt_token(token):\n    # performs jwt checks\n    return True",
            language="python",
            function_name="verify_jwt_token",
            chunk_index=0,
            start_line=1,
            end_line=3
        ),
        CodeChunk(
            file_path="db.py",
            repo_name=repo,
            content="def connect_to_database():\n    # postgres connection\n    return db",
            language="python",
            function_name="connect_to_database",
            chunk_index=0,
            start_line=1,
            end_line=3
        )
    ]
    
    # Store chunks
    temp_vector_store.add_chunks(repo, chunks)
    
    # Semantic Search
    results = temp_vector_store.search(repo, "postgres connection", top_k=1)
    
    assert len(results) == 1
    assert results[0].file_path == "db.py"
    assert results[0].function_name == "connect_to_database"
    assert "postgres connection" in results[0].content
