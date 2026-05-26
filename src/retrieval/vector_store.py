import os
import logging
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.models.schemas import CodeChunk

logger = logging.getLogger(__name__)

class CodeVectorStore:
    def __init__(
        self, 
        persist_directory: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB client in {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        logger.info(f"Loading sentence transformer embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

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
        # Local batch embedding
        chunk_embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
        embeddings = [emb.tolist() for emb in chunk_embeddings]
        
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
        query_embedding = self.embedding_model.encode(query).tolist()
        
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
