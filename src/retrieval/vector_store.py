import os
import logging
from typing import List, Optional
import chromadb
from chromadb.config import Settings

from src.models.schemas import CodeChunk

logger = logging.getLogger(__name__)

class CodeVectorStore:
    def __init__(
        self, 
        persist_directory: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_provider: Optional[str] = None
    ):
        self.embedding_provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", "local")
        
        # Determine default model name based on provider
        if not model_name:
            if self.embedding_provider == "local":
                self.model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            elif self.embedding_provider == "openai":
                self.model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            elif self.embedding_provider == "cohere":
                self.model_name = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")
            elif self.embedding_provider == "huggingface":
                self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            else:
                self.model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        else:
            self.model_name = model_name

        self.embedding_model = None
        if self.embedding_provider == "local":
            logger.info(f"Loading local sentence transformer embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.model_name)
        else:
            logger.info(f"Using remote embedding provider '{self.embedding_provider}' with model: {self.model_name}")

        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_host = os.getenv("CHROMA_HOST")
        
        if chroma_tenant:
            chroma_api_key = os.getenv("CHROMA_API_KEY")
            chroma_database = os.getenv("CHROMA_DATABASE", "default")
            logger.info(f"Initializing managed Chroma Cloud client (Tenant: {chroma_tenant}, Database: {chroma_database})")
            self.client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database
            )
        elif chroma_host:
            chroma_port = os.getenv("CHROMA_PORT", "8000")
            chroma_ssl = os.getenv("CHROMA_SSL", "false").lower() == "true"
            chroma_token = os.getenv("CHROMA_API_KEY") or os.getenv("CHROMA_AUTH_TOKEN")
            
            logger.info(f"Initializing remote ChromaDB client at {chroma_host}:{chroma_port} (SSL: {chroma_ssl})")
            
            headers = {}
            settings_kwargs = {"anonymized_telemetry": False}
            if chroma_token:
                headers["Authorization"] = f"Bearer {chroma_token}"
                headers["X-Chroma-Token"] = chroma_token
                settings_kwargs["chroma_client_auth_provider"] = "chromadb.auth.token.TokenAuthClientProvider"
                settings_kwargs["chroma_client_auth_credentials"] = chroma_token
                
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                ssl=chroma_ssl,
                headers=headers,
                settings=Settings(**settings_kwargs)
            )
        else:
            self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Initializing local ChromaDB client in {self.persist_directory}")
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

    def _embed_documents(self, documents: List[str], input_type: str = "search_document") -> List[List[float]]:
        if not documents:
            return []
            
        provider = self.embedding_provider.lower()
        if provider == "local":
            if not self.embedding_model:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.model_name)
            chunk_embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
            return [emb.tolist() for emb in chunk_embeddings]
            
        elif provider == "openai":
            import httpx
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required when using the 'openai' embedding provider.")
            
            headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "input": documents,
                "model": self.model_name
            }
            response = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
            
        elif provider == "cohere":
            import httpx
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                raise ValueError("COHERE_API_KEY environment variable is required when using the 'cohere' embedding provider.")
            
            headers = {
                "Authorization": f"Bearer {cohere_api_key}",
                "Content-Type": "application/json"
            }
            cohere_input_type = "search_document" if input_type == "search_document" else "search_query"
            payload = {
                "texts": documents,
                "model": self.model_name,
                "input_type": cohere_input_type
            }
            response = httpx.post(
                "https://api.cohere.ai/v1/embed",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, dict):
                return embeddings.get("float", [])
            return embeddings
            
        elif provider == "huggingface":
            import httpx
            hf_api_key = os.getenv("HF_API_KEY")
            headers = {}
            if hf_api_key:
                headers["Authorization"] = f"Bearer {hf_api_key}"
            
            response = httpx.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}",
                headers=headers,
                json={"inputs": documents},
                timeout=60.0
            )
            response.raise_for_status()
            embeddings = response.json()
            if isinstance(embeddings, list) and len(embeddings) > 0:
                if isinstance(embeddings[0], float):
                    return [embeddings]
                return embeddings
            raise ValueError(f"Unexpected response structure from Hugging Face Inference API: {embeddings}")
            
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")

    def _get_collection(self, repo_name: str):
        # Format repo_name to be a valid collection name (replace / and other symbols)
        collection_name = repo_name.replace("/", "_").replace("-", "_").lower()
        # Keep name within length limits of ChromaDB collections (3-63 chars)
        if len(collection_name) > 63:
            collection_name = collection_name[:63]
        return self.client.get_or_create_collection(collection_name)

    def add_chunks(self, repo_name: str, chunks: List[CodeChunk]):
        """Embed and store chunks in the repo-specific collection."""
        if not chunks:
            logger.warning("No chunks to add to vector store.")
            return
            
        collection = self._get_collection(repo_name)
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            # Construct a unique ID for each chunk
            chunk_id = f"{chunk.file_path}#chunk{chunk.chunk_index}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            
            # Metadata for filters/citations
            metadata = {
                "file_path": chunk.file_path,
                "repo_name": chunk.repo_name,
                "language": chunk.language,
                "chunk_index": chunk.chunk_index,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line
            }
            if chunk.function_name:
                metadata["function_name"] = chunk.function_name
            if chunk.class_name:
                metadata["class_name"] = chunk.class_name
                
            metadatas.append(metadata)
            
        logger.info(f"Embedding {len(documents)} chunks...")
        embeddings = self._embed_documents(documents, input_type="search_document")
        
        logger.info(f"Adding chunks to Chroma collection for {repo_name}...")
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Added {len(chunks)} chunks successfully.")

    def search(self, repo_name: str, query: str, top_k: int = 20) -> List[CodeChunk]:
        """Perform semantic search using locally generated embeddings."""
        collection = self._get_collection(repo_name)
        
        # Check if collection is empty
        if collection.count() == 0:
            logger.warning(f"Chroma collection for {repo_name} is empty.")
            return []
            
        # Encode the query
        query_embedding = self._embed_documents([query], input_type="search_query")[0]
        
        # Search
        results = collection.query(

            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        chunks = []
        if results and results["documents"] and len(results["documents"][0]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            
            for doc, meta in zip(docs, metas):
                chunks.append(CodeChunk(
                    file_path=meta["file_path"],
                    repo_name=meta["repo_name"],
                    content=doc,
                    language=meta["language"],
                    function_name=meta.get("function_name"),
                    class_name=meta.get("class_name"),
                    chunk_index=int(meta["chunk_index"]),
                    start_line=int(meta["start_line"]),
                    end_line=int(meta["end_line"])
                ))
                
        return chunks

    def list_repos(self) -> List[str]:
        """List all ingested repositories by listing Chroma collection names."""
        collections = self.client.list_collections()
        # Map them back if needed, or just return collections names
        return [c.name for c in collections]

    def delete_repo(self, repo_name: str):
        """Delete collection for a specific repo."""
        collection_name = repo_name.replace("/", "_").replace("-", "_").lower()
        if len(collection_name) > 63:
            collection_name = collection_name[:63]
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
