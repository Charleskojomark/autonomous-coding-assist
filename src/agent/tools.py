import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

from src.retrieval.vector_store import CodeVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.ingestion.github_client import GithubIngestionClient

logger = logging.getLogger(__name__)

# Singletons/shared instances that are initialized at runtime
_vector_store: Optional[CodeVectorStore] = None
_reranker: Optional[CrossEncoderReranker] = None
_github_client: Optional[GithubIngestionClient] = None

def init_tools(vector_store: CodeVectorStore, reranker: CrossEncoderReranker, github_client: GithubIngestionClient):
    global _vector_store, _reranker, _github_client
    _vector_store = vector_store
    _reranker = reranker
    _github_client = github_client

@tool
def search_codebase(query: str, repo_name: str) -> str:
    """
    Search the codebase semantically for code chunks relevant to the query.
    Returns the top reranked chunks with code contents and line ranges.
    
    Args:
        query: The search query (e.g. "JWT token validation" or "database connection pooling")
        repo_name: The name of the repository in the format "owner/repo"
    """
    if not _vector_store or not _reranker:
        return "Error: Retrieval system not initialized."
        
    logger.info(f"Agent tool search_codebase called for '{repo_name}' with query: '{query}'")
    try:
        # Retrieve candidate chunks (top 20)
        candidates = _vector_store.search(repo_name, query, top_k=20)
        if not candidates:
            return f"No results found in vector store for repo '{repo_name}'."
            
        # Rerank to top 5
        reranked = _reranker.rerank(query, candidates, top_n=5)
        
        # Format output for agent context
        formatted_results = []
        for idx, chunk in enumerate(reranked):
            citation_str = f"File: {chunk.file_path} (Lines: {chunk.start_line}-{chunk.end_line}"
            if chunk.class_name:
                citation_str += f", Class: {chunk.class_name}"
            if chunk.function_name:
                citation_str += f", Function: {chunk.function_name}"
            citation_str += ")"
            
            chunk_repr = f"--- Result #{idx+1} | {citation_str} ---\n```{chunk.language}\n{chunk.content}\n```\n"
            formatted_results.append(chunk_repr)
            
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in search_codebase tool: {e}")
        return f"Error executing semantic search: {str(e)}"

@tool
def get_file(path: str, repo_name: str) -> str:
    """
    Fetch the full content of a specific file from the repository via GitHub API.
    Use this when you have found a relevant file chunk and want to inspect the entire file context.
    
    Args:
        path: The relative file path in the repo (e.g. "src/auth/middleware.py")
        repo_name: The name of the repository in the format "owner/repo"
    """
    if not _github_client:
        return "Error: GitHub client not initialized."
        
    logger.info(f"Agent tool get_file called for '{repo_name}' at path '{path}'")
    try:
        repo = _github_client.get_repo(repo_name)
        content = _github_client.fetch_file_content(repo, path)
        return f"--- Full Content of file: {path} ---\n```\n{content}\n```"
    except Exception as e:
        logger.error(f"Error in get_file tool for {path}: {e}")
        return f"Error fetching file '{path}': {str(e)}. Make sure the path is correct and exists."

@tool
def get_issues(repo_name: str) -> str:
    """
    Fetch open issues from the repository to look for bugs or user problems.
    
    Args:
        repo_name: The name of the repository in the format "owner/repo"
    """
    if not _github_client:
        return "Error: GitHub client not initialized."
        
    logger.info(f"Agent tool get_issues called for '{repo_name}'")
    try:
        issues = _github_client.fetch_open_issues(repo_name)
        if not issues:
            return f"No open issues found in repo '{repo_name}'."
            
        formatted_issues = []
        for issue in issues[:10]: # limit to top 10
            issue_repr = f"Issue #{issue['number']}: {issue['title']}\nURL: {issue['url']}\nBody: {issue['body'][:300]}...\n"
            formatted_issues.append(issue_repr)
            
        return "\n---\n".join(formatted_issues)
    except Exception as e:
        logger.error(f"Error in get_issues tool: {e}")
        return f"Error fetching issues: {str(e)}"
