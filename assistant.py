import os
import sys
import typer
import logging
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()

# Silence annoying libraries logs
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("hnswlib").setLevel(logging.WARNING)

# Add current directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.github_client import GithubIngestionClient
from src.ingestion.chunker import CodeChunker
from src.retrieval.vector_store import CodeVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.agent.tools import init_tools
from src.agent.graph import create_agent_graph
from src.agent.issue_analyzer import IssueAnalyzer

app = typer.Typer(help="Autonomous Coding Assistant - Q&A and Bug-Fix Agent")
console = Console()

@app.command()
def ask(
    query: str = typer.Argument(..., help="The question or search query"),
    repo: str = typer.Option(..., "--repo", "-r", help="Repository in 'owner/repo' format"),
    mode: str = typer.Option("qa", "--mode", "-m", help="Mode: 'qa' for codebase Q&A, 'issue' for structured bug-fixing"),
    issue_number: Optional[int] = typer.Option(None, "--issue-number", "-i", help="GitHub issue number (required for issue mode)")
):
    """
    Query the codebase assistant or analyze open issues for bug-fix suggestions.
    """
    if "/" not in repo:
        console.print("[bold red]Error: Repo name must be in the format 'owner/repo'[/bold red]")
        raise typer.Exit(code=1)
        
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        console.print("[bold red]Error: GROQ_API_KEY environment variable not set. Please set it in your .env file.[/bold red]")
        raise typer.Exit(code=1)

    # Initialize Core Services
    github_client = GithubIngestionClient()
    vector_store = CodeVectorStore()
    
    # Simple check to see if repo has any vectors
    col_name = repo.replace("/", "_").replace("-", "_").lower()[:63]
    try:
        col = vector_store.client.get_collection(col_name)
        if col.count() == 0:
            console.print(f"[bold red]Error: Repository '{repo}' is not ingested yet. Run: python ingest.py --repo {repo}[/bold red]")
            raise typer.Exit(code=1)
    except Exception:
        console.print(f"[bold red]Error: Repository '{repo}' is not ingested yet. Run: python ingest.py --repo {repo}[/bold red]")
        raise typer.Exit(code=1)

    reranker = CrossEncoderReranker()
    init_tools(vector_store, reranker, github_client)

    if mode == "qa":
        # Q&A Mode using LangGraph Agent Graph
        agent_graph = create_agent_graph()
        
        console.print(f"[bold blue]🤖 Coding Assistant thinking about your query in '{repo}'...[/bold blue]\n")
        
        from langchain_core.messages import HumanMessage
        state = {
            "messages": [HumanMessage(content=query)],
            "repo_name": repo
        }
        
        try:
            output = agent_graph.invoke(state)
            final_message = output["messages"][-1].content
            
            # Print response inside a panel
            console.print(Panel(Markdown(final_message), title="🤖 Assistant Answer", border_style="green"))
            
        except Exception as e:
            console.print(f"[bold red]❌ Agent Graph Execution failed: {str(e)}[/bold red]")
            raise typer.Exit(code=1)
            
    elif mode == "issue":
        # Structured Bug Fixing Mode
        if not issue_number:
            console.print("[bold red]Error: --issue-number (-i) is required when using 'issue' mode.[/bold red]")
            raise typer.Exit(code=1)
            
        issue_analyzer = IssueAnalyzer(vector_store, reranker, github_client)
        
        console.print(f"[bold blue]🔍 Finding root cause and generating bug-fix suggestions for Issue #{issue_number} in '{repo}'...[/bold blue]\n")
        
        try:
            fix = issue_analyzer.analyze_and_suggest_fix(repo, issue_number)
            
            # Display proposed fix
            console.print(Panel(f"[bold]Root Cause Hypothesis:[/bold]\n{fix.root_cause_hypothesis}", title="🔍 Root Cause Hypothesis", border_style="cyan"))
            
            func_str = f" -> function '{fix.responsible_function}'" if fix.responsible_function else ""
            console.print(Panel(
                f"[bold]File:[/bold] {fix.responsible_file}{func_str}\n"
                f"[bold]Confidence Score:[/bold] {fix.confidence_score * 100:.1f}%\n\n"
                f"[bold]Suggested Diff:[/bold]\n{fix.suggested_change}\n\n"
                f"[bold]Explanation:[/bold]\n{fix.explanation}",
                title="🛠 Suggested Fix",
                border_style="green"
            ))
            
            console.print(Panel(fix.test_case, title="🧪 Verification Test Case", border_style="magenta"))
            
        except Exception as e:
            console.print(f"[bold red]❌ Issue analysis failed: {str(e)}[/bold red]")
            raise typer.Exit(code=1)
    else:
        console.print(f"[bold red]Error: Unknown mode '{mode}'. Choose 'qa' or 'issue'.[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
