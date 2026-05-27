import os
import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.vector_store import CodeVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.models.schemas import CodeChunk

def test_chroma_remote_client_init():
    with patch("chromadb.HttpClient") as mock_http_client:
        env = {
            "CHROMA_HOST": "example.com",
            "CHROMA_PORT": "8080",
            "CHROMA_SSL": "true",
            "CHROMA_AUTH_TOKEN": "my-secret-token",
            "EMBEDDING_PROVIDER": "openai"
        }
        with patch.dict(os.environ, env):
            store = CodeVectorStore(persist_directory=None, model_name="text-embedding-3-small")
            
            mock_http_client.assert_called_once()
            args, kwargs = mock_http_client.call_args
            assert kwargs["host"] == "example.com"
            assert kwargs["port"] == "8080"
            assert kwargs["ssl"] is True
            assert kwargs["headers"] == {
                "Authorization": "Bearer my-secret-token",
                "X-Chroma-Token": "my-secret-token"
            }
            assert kwargs["settings"].chroma_client_auth_provider == "chromadb.auth.token.TokenAuthClientProvider"
            assert kwargs["settings"].chroma_client_auth_credentials == "my-secret-token"

def test_openai_embedding_provider():
    with patch("chromadb.PersistentClient") as mock_persistent_client:
        with patch("httpx.post") as mock_post:
            # Mock OpenAI API response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            mock_post.return_value = mock_response
            
            env = {
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-3-small",
                "OPENAI_API_KEY": "sk-test"
            }
            with patch.dict(os.environ, env):
                store = CodeVectorStore(persist_directory="/tmp/dummy")
                embeddings = store._embed_documents(["hello"])
                
                assert embeddings == [[0.1, 0.2, 0.3]]
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert args[0] == "https://api.openai.com/v1/embeddings"
                assert kwargs["headers"]["Authorization"] == "Bearer sk-test"

def test_cohere_embedding_provider():
    with patch("chromadb.PersistentClient") as mock_persistent_client:
        with patch("httpx.post") as mock_post:
            # Mock Cohere API response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embeddings": [[0.4, 0.5, 0.6]]
            }
            mock_post.return_value = mock_response
            
            env = {
                "EMBEDDING_PROVIDER": "cohere",
                "EMBEDDING_MODEL": "embed-english-v3.0",
                "COHERE_API_KEY": "cohere-test"
            }
            with patch.dict(os.environ, env):
                store = CodeVectorStore(persist_directory="/tmp/dummy")
                embeddings = store._embed_documents(["hello"], input_type="search_query")
                
                assert embeddings == [[0.4, 0.5, 0.6]]
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert args[0] == "https://api.cohere.ai/v1/embed"
                assert kwargs["headers"]["Authorization"] == "Bearer cohere-test"
                assert kwargs["json"]["input_type"] == "search_query"

def test_hf_embedding_provider():
    with patch("chromadb.PersistentClient") as mock_persistent_client:
        with patch("httpx.post") as mock_post:
            # Mock HF API response
            mock_response = MagicMock()
            mock_response.json.return_value = [[0.7, 0.8, 0.9]]
            mock_post.return_value = mock_response
            
            env = {
                "EMBEDDING_PROVIDER": "huggingface",
                "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
                "HF_API_KEY": "hf-test"
            }
            with patch.dict(os.environ, env):
                store = CodeVectorStore(persist_directory="/tmp/dummy")
                embeddings = store._embed_documents(["hello"])
                
                assert embeddings == [[0.7, 0.8, 0.9]]
                mock_post.assert_called_once()
                args, kwargs = mock_post.call_args
                assert args[0] == "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
                assert kwargs["headers"]["Authorization"] == "Bearer hf-test"

def test_cohere_reranker():
    with patch("httpx.post") as mock_post:
        # Mock Cohere Rerank API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1}
            ]
        }
        mock_post.return_value = mock_response
        
        env = {
            "RERANKER_PROVIDER": "cohere",
            "RERANKER_MODEL": "rerank-english-v3.0",
            "COHERE_API_KEY": "cohere-test"
        }
        with patch.dict(os.environ, env):
            reranker = CrossEncoderReranker()
            chunks = [
                CodeChunk(file_path="1.py", repo_name="a", content="first", language="py", chunk_index=0, start_line=1, end_line=2),
                CodeChunk(file_path="2.py", repo_name="a", content="second", language="py", chunk_index=1, start_line=1, end_line=2)
            ]
            reranked = reranker.rerank("query", chunks)
            
            assert len(reranked) == 2
            assert reranked[0].file_path == "2.py"
            assert reranked[1].file_path == "1.py"
            mock_post.assert_called_once()

def test_hf_reranker():
    with patch("httpx.post") as mock_post:
        # Mock HF Inference API response (list of floats)
        mock_response = MagicMock()
        mock_response.json.return_value = [0.1, 0.9]
        mock_post.return_value = mock_response
        
        env = {
            "RERANKER_PROVIDER": "huggingface",
            "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "HF_API_KEY": "hf-test"
        }
        with patch.dict(os.environ, env):
            reranker = CrossEncoderReranker()
            chunks = [
                CodeChunk(file_path="1.py", repo_name="a", content="first", language="py", chunk_index=0, start_line=1, end_line=2),
                CodeChunk(file_path="2.py", repo_name="a", content="second", language="py", chunk_index=1, start_line=1, end_line=2)
            ]
            reranked = reranker.rerank("query", chunks)
            
            assert len(reranked) == 2
            assert reranked[0].file_path == "2.py"
            assert reranked[1].file_path == "1.py"
            mock_post.assert_called_once()

def test_chroma_cloud_client_init():
    with patch("chromadb.CloudClient") as mock_cloud_client:
        env = {
            "CHROMA_TENANT": "e73b3210-08f1-4d87-91b2-e08cdddbbdd1",
            "CHROMA_DATABASE": "mychromadb",
            "CHROMA_API_KEY": "ck-HwpGM6CSuQdTrR1MBUkSmfUEurZ9u9ktGUYsdakehYV7",
            "EMBEDDING_PROVIDER": "openai"
        }
        with patch.dict(os.environ, env):
            store = CodeVectorStore(persist_directory=None, model_name="text-embedding-3-small")
            
            mock_cloud_client.assert_called_once_with(
                api_key="ck-HwpGM6CSuQdTrR1MBUkSmfUEurZ9u9ktGUYsdakehYV7",
                tenant="e73b3210-08f1-4d87-91b2-e08cdddbbdd1",
                database="mychromadb"
            )

