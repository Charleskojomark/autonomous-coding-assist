import os
import sys
import typer
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv

load_dotenv()

# Add current directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.github_client import GithubIngestionClient
from src.ingestion.chunker import CodeChunker
from src.retrieval.vector_store import CodeVectorStore

app = typer.Typer(help="Autonomous Coding Assistant Ingestion Tool")
console = Console()

@app.command()
def ingest(
    repo: str = typer.Option(..., "--repo", "-r", help="Repository in 'owner/repo' format"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Git branch to ingest"),
    extensions: List[str] = typer.Option(
        [".py", ".js", ".ts", ".md"], 
        "--ext", 
        "-e", 
        help="File extensions to ingest"
    )
):
    """
    Ingest all files from a GitHub repository, split them into code-aware chunks, and store them in ChromaDB.
    """
    console.print(f"[bold blue]🚀 Starting ingestion for repository: {repo}[/bold blue]")
    
    if "/" not in repo:
        console.print("[bold red]Error: Repo name must be in the format 'owner/repo'[/bold red]")
        raise typer.Exit(code=1)

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        console.print("[yellow]Warning: GITHUB_TOKEN not set. API rate limits will be highly restricted.[/yellow]")
        
    try:
        # Initialize clients
        gh_client = GithubIngestionClient(token=github_token)
        chunker = CodeChunker()
        vector_store = CodeVectorStore()
        
        ext_set = set(extensions)
        
        # Phase 1: Fetching Repository
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            progress.add_task(description=f"Fetching repository tree for '{repo}'...", total=None)
            raw_files = gh_client.fetch_repo_files(repo, ext_set, branch)
            
        console.print(f"[bold green]✓ Fetched {len(raw_files)} files successfully.[/bold green]")
        
        if not raw_files:
            console.print("[yellow]No files matched the specified extensions. Ingestion skipped.[/yellow]")
            return
            
        # Phase 2: Chunking Files
        all_chunks = []
        with Progress(
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(description="Splitting files into code-aware chunks...", total=len(raw_files))
            for raw_file in raw_files:
                chunks = chunker.chunk_file(raw_file, repo)
                all_chunks.extend(chunks)
                progress.advance(task)
                
        console.print(f"[bold green]✓ Created {len(all_chunks)} code chunks with AST metadata.[/bold green]")
        
        # Phase 3: Storing in Vector Store
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            progress.add_task(description="Generating embeddings and saving to ChromaDB...", total=None)
            vector_store.add_chunks(repo, all_chunks)
            
        console.print(f"[bold green]🎉 Ingestion completed successfully for {repo}! Chunks stored in ChromaDB.[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Ingestion failed: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
