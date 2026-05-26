import os
import logging
from typing import List
from sentence_transformers import CrossEncoder

from src.models.schemas import CodeChunk

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading CrossEncoder reranker model: {model_name}")
        # Can run on GPU if available, falls back to CPU
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[CodeChunk], top_n: int = 5) -> List[CodeChunk]:
        """
        Rerank a list of retrieved chunks using a cross-encoder model.
        Returns top_n elements sorted by descending relevance.
        """
        if not chunks:
            return []
            
        logger.info(f"Reranking {len(chunks)} candidate chunks for query: '{query}'...")
        
        # Prepare pairs: (query, text)
        pairs = [(query, chunk.content) for chunk in chunks]
        
        # Compute relevance scores
        scores = self.model.predict(pairs)
        
        # Zip chunks with scores and sort descending
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Print top ranked file paths and scores for debugging/logs
        logger.info("Top reranked results:")
        for idx, (chunk, score) in enumerate(scored_chunks[:top_n]):
            func_str = f" in {chunk.function_name}" if chunk.function_name else ""
            logger.info(f"  #{idx+1}: {chunk.file_path}{func_str} | Score: {score:.4f}")
            
        # Return top N chunks
        return [chunk for chunk, score in scored_chunks[:top_n]]
