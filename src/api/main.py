import os
import logging
from typing import List, Optional, Dict, Any, Set
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from src.models.schemas import IssueFix, AgentResponse, Citation
from src.ingestion.github_client import GithubIngestionClient
from src.ingestion.chunker import CodeChunker
from src.retrieval.vector_store import CodeVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.agent.tools import init_tools
from src.agent.graph import create_agent_graph
from src.agent.issue_analyzer import IssueAnalyzer

app = FastAPI(
    title="Autonomous Coding Assistant API",
    description="Production-grade API for repository ingestion, code-aware RAG, and stateful multi-tool agent Q&A",
    version="0.1.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
github_client = GithubIngestionClient()
chunker = CodeChunker()
vector_store = CodeVectorStore()
reranker = CrossEncoderReranker()

# Initialize Agent Tools
init_tools(vector_store, reranker, github_client)
agent_graph = create_agent_graph()
issue_analyzer = IssueAnalyzer(vector_store, reranker, github_client)

# Pydantic schemas for Request/Response bodies
class IngestRequest(BaseModel):
    repo: str
    extensions: Optional[List[str]] = [".py", ".js", ".ts", ".md"]
    branch: Optional[str] = None

class QueryRequest(BaseModel):
    repo: str
    query: str
    history: Optional[List[Dict[str, Any]]] = []

class AnalyzeIssueRequest(BaseModel):
    repo: str
    issue_number: int

# Active ingestion tasks status
ingestion_status: Dict[str, str] = {}

def bg_ingest_repo(repo_name: str, extensions: Set[str], branch: Optional[str] = None):
    try:
        ingestion_status[repo_name] = "Fetching files from GitHub..."
        raw_files = github_client.fetch_repo_files(repo_name, extensions, branch)
        
        ingestion_status[repo_name] = f"Splitting {len(raw_files)} files into code chunks..."
        all_chunks = []
        for raw_file in raw_files:
            chunks = chunker.chunk_file(raw_file, repo_name)
            all_chunks.extend(chunks)
            
        ingestion_status[repo_name] = f"Storing {len(all_chunks)} chunks in vector database..."
        vector_store.add_chunks(repo_name, all_chunks)
        
        ingestion_status[repo_name] = "Completed"
        logger.info(f"Ingestion for {repo_name} finished successfully!")
    except Exception as e:
        logger.error(f"Failed to ingest repo {repo_name}: {e}")
        ingestion_status[repo_name] = f"Failed: {str(e)}"

@app.post("/ingest", summary="Ingest a GitHub repository")
def ingest_repository(request: IngestRequest, background_tasks: BackgroundTasks):
    repo = request.repo
    if not repo or "/" not in repo:
        raise HTTPException(status_code=400, detail="Invalid repository name. Format must be 'owner/repo'")
        
    ext_set = set(request.extensions)
    ingestion_status[repo] = "Queued"
    
    background_tasks.add_task(bg_ingest_repo, repo, ext_set, request.branch)
    
    return {
        "status": "Queued",
        "message": f"Ingestion of {repo} has been started in the background.",
        "repo": repo
    }

@app.get("/ingest/status/{owner}/{repo}", summary="Check repository ingestion status")
def get_ingestion_status(owner: str, repo: str):
    repo_name = f"{owner}/{repo}"
    status = ingestion_status.get(repo_name, "Not Found")
    return {"repo": repo_name, "status": status}

@app.get("/repos", summary="List all ingested repositories")
def list_repositories():
    repos = vector_store.list_repos()
    return {"repositories": repos}

@app.post("/query", response_model=AgentResponse, summary="Query the codebase using LangGraph agent")
def query_codebase(request: QueryRequest):
    repo = request.repo
    query = request.query
    
    # Simple check to see if repo has any vectors
    col_name = repo.replace("/", "_").replace("-", "_").lower()[:63]
    try:
        col = vector_store.client.get_collection(col_name)
        if col.count() == 0:
            raise HTTPException(status_code=404, detail=f"Repository '{repo}' has no ingested code. Please ingest it first.")
    except Exception:
        raise HTTPException(status_code=404, detail=f"Repository '{repo}' is not ingested. Please run ingestion first.")

    # Formulate inputs for the LangGraph agent
    from langchain_core.messages import HumanMessage, AIMessage
    
    messages = []
    # Feed message history if provided
    for msg in request.history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content")))
            
    messages.append(HumanMessage(content=query))
    
    state = {
        "messages": messages,
        "repo_name": repo
    }
    
    try:
        logger.info(f"Running LangGraph agent query for repo '{repo}'")
        output = agent_graph.invoke(state)
        
        # Format citations from vector search / chunk metadata if any
        # The agent response is the last message in state
        final_message = output["messages"][-1].content
        
        # Simple extraction of citations from the final text to build structured citations
        citations = []
        # Finding patterns like `path/to/file` (lines X-Y, function `func`)
        citation_matches = re.findall(r'`([^`\n]+)`\s*\(lines\s+(\d+)-(\d+)(?:,\s+function\s+`([^`\n]+)`)?(?:,\s+class\s+`([^`\n]+)`)?\)', final_message)
        for match in citation_matches:
            citations.append(Citation(
                file_path=match[0],
                start_line=int(match[1]),
                end_line=int(match[2]),
                function_name=match[3] if match[3] else None,
                class_name=match[4] if match[4] else None
            ))
            
        return AgentResponse(
            response=final_message,
            citations=citations
        )
    except Exception as e:
        logger.error(f"Error executing agent graph: {e}")
        raise HTTPException(status_code=500, detail=f"Agent workflow failed: {str(e)}")

@app.post("/analyze-issue", response_model=IssueFix, summary="Provide Issue Fix Suggestions")
def analyze_issue(request: AnalyzeIssueRequest):
    repo = request.repo
    issue_num = request.issue_number
    
    try:
        fix = issue_analyzer.analyze_and_suggest_fix(repo, issue_num)
        return fix
    except Exception as e:
        logger.error(f"Error during issue analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Issue analysis failed: {str(e)}")

import re
