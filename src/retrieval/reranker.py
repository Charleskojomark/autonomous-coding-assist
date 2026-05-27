import os
import logging
from typing import List, Optional

from src.models.schemas import CodeChunk

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(
        self, 
        model_name: Optional[str] = None,
        reranker_provider: Optional[str] = None
    ):
        self.reranker_provider = reranker_provider or os.getenv("RERANKER_PROVIDER", "local")
        
        # Determine default model name based on provider
        if not model_name:
            if self.reranker_provider == "local":
                self.model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            elif self.reranker_provider == "cohere":
                self.model_name = os.getenv("RERANKER_MODEL", "rerank-english-v3.0")
            elif self.reranker_provider == "huggingface":
                self.model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            else:
                self.model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            self.model_name = model_name

        self.model = None
        if self.reranker_provider == "local":
            logger.info(f"Loading local CrossEncoder reranker model: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        else:
            logger.info(f"Using remote reranker provider '{self.reranker_provider}' with model: {self.model_name}")

    def rerank(self, query: str, chunks: List[CodeChunk], top_n: int = 5) -> List[CodeChunk]:
        """
        Rerank a list of retrieved chunks using a cross-encoder model/API.
        Returns top_n elements sorted by descending relevance.
        """
        if not chunks:
            return []
            
        provider = self.reranker_provider.lower()
        
        if provider == "local":
            if not self.model:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
            logger.info(f"Reranking {len(chunks)} candidate chunks for query: '{query}' locally...")
            pairs = [(query, chunk.content) for chunk in chunks]
            scores = self.model.predict(pairs)
            scored_chunks = list(zip(chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
        elif provider == "cohere":
            import httpx
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                raise ValueError("COHERE_API_KEY environment variable is required when using the 'cohere' reranker.")
                
            logger.info(f"Reranking {len(chunks)} candidate chunks for query: '{query}' via Cohere Rerank API...")
            headers = {
                "Authorization": f"Bearer {cohere_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": [c.content for c in chunks],
                "top_n": top_n
            }
            response = httpx.post(
                "https://api.cohere.ai/v1/rerank",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            
            # Map index back to chunks
            reranked = []
            for item in results:
                idx = item["index"]
                reranked.append(chunks[idx])
            return reranked[:top_n]
            
        elif provider == "huggingface":
            import httpx
            hf_api_key = os.getenv("HF_API_KEY")
            headers = {}
            if hf_api_key:
                headers["Authorization"] = f"Bearer {hf_api_key}"
                
            logger.info(f"Reranking {len(chunks)} candidate chunks for query: '{query}' via Hugging Face Inference API...")
            payload = [{"text": query, "text_pair": chunk.content} for chunk in chunks]
            
            response = httpx.post(
                f"https://router.huggingface.co/hf-inference/models/{self.model_name}",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            scores = response.json()
            
            # Extract scores and map to chunks
            if isinstance(scores, list) and len(scores) > 0:
                if all(isinstance(s, (int, float)) for s in scores):
                    scored_chunks = list(zip(chunks, scores))
                elif all(isinstance(s, dict) for s in scores):
                    scored_chunks = list(zip(chunks, [s.get("score", 0.0) for s in scores]))
                elif all(isinstance(s, list) for s in scores):
                    extracted_scores = []
                    for item in scores:
                        if isinstance(item, list) and len(item) > 0:
                            # Try to extract the score for standard binary cross-encoder labels
                            item_score = next((x["score"] for x in item if x.get("label") in ("LABEL_1", "LABEL_0")), item[0].get("score", 0.0))
                            extracted_scores.append(item_score)
                        else:
                            extracted_scores.append(0.0)
                    scored_chunks = list(zip(chunks, extracted_scores))
                else:
                    scored_chunks = list(zip(chunks, [0.0] * len(chunks)))
            else:
                scored_chunks = list(zip(chunks, [0.0] * len(chunks)))
                
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
        else:
            raise ValueError(f"Unknown reranker provider: {self.reranker_provider}")

        # Print top ranked file paths and scores for debugging/logs
        logger.info("Top reranked results:")
        for idx, (chunk, score) in enumerate(scored_chunks[:top_n]):
            func_str = f" in {chunk.function_name}" if chunk.function_name else ""
            logger.info(f"  #{idx+1}: {chunk.file_path}{func_str} | Score: {score:.4f}")
            
        # Return top N chunks
        return [chunk for chunk, score in scored_chunks[:top_n]]

