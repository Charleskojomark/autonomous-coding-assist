import logging
import base64
import os
from typing import List, Optional, Set, Dict, Any
from datetime import datetime
from github import Github, GithubException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models.schemas import RawFile

logger = logging.getLogger(__name__)

# Map file extensions to languages
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
}

class GithubIngestionClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            logger.warning("No GITHUB_TOKEN found. API rate limits will be extremely restricted.")
        self.gh = Github(self.token)

    @retry(
        retry=retry_if_exception_type(GithubException),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def get_repo(self, repo_name: str):
        """Get the repository object, retrying on rate limit or temp errors."""
        try:
            return self.gh.get_repo(repo_name)
        except GithubException as e:
            if e.status == 403 or e.status == 429:
                logger.warning(f"Rate limited or forbidden. Retrying get_repo for {repo_name}...")
            raise e

    @retry(
        retry=retry_if_exception_type(GithubException),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def fetch_file_content(self, repo, path: str, ref: str = "main") -> str:
        """Fetch content for a specific file path from Github with rate limit handling."""
        try:
            content_file = repo.get_contents(path, ref=ref)
            if isinstance(content_file, list):
                raise ValueError(f"Path {path} is a directory, not a file.")
            
            # GitHub returns base64 encoded content for files, let's decode it
            if content_file.encoding == "base64":
                return base64.b64decode(content_file.content).decode("utf-8", errors="ignore")
            else:
                return content_file.decoded_content.decode("utf-8", errors="ignore")
        except GithubException as e:
            if e.status == 403 or e.status == 429:
                logger.warning(f"Rate limited or forbidden. Retrying fetch_file_content for {path}...")
            raise e

    def fetch_repo_files(
        self, 
        repo_name: str, 
        extensions: Optional[Set[str]] = None,
        branch: Optional[str] = None
    ) -> List[RawFile]:
        """
        Fetch all files from a repository filtering by extension.
        Uses recursive tree API to avoid fetching dir contents recursively one by one.
        """
        if extensions is None:
            extensions = {".py", ".js", ".ts", ".md", ".json"}
        
        logger.info(f"Ingesting repository: {repo_name}")
        repo = self.get_repo(repo_name)
        
        # Determine default branch if not specified
        if not branch:
            branch = repo.default_branch
        
        logger.info(f"Using branch: {branch}")
        
        # Fetch recursive tree
        try:
            ref = repo.get_branch(branch)
            sha = ref.commit.sha
            tree = repo.get_git_tree(sha, recursive=True)
        except GithubException as e:
            logger.error(f"Failed to fetch git tree for {repo_name}: {e}")
            raise e
            
        raw_files = []
        
        for element in tree.tree:
            # Type 'blob' is a file, 'tree' is a directory
            if element.type == "blob":
                path = element.path
                _, ext = os.path.splitext(path.lower())
                
                if ext in extensions:
                    logger.info(f"Queueing file for fetching: {path}")
                    try:
                        content = self.fetch_file_content(repo, path, ref=sha)
                        language = EXTENSION_TO_LANGUAGE.get(ext, "text")
                        
                        raw_file = RawFile(
                            path=path,
                            content=content,
                            size=element.size or len(content),
                            last_modified=datetime.utcnow(), # fallback
                            language=language
                        )
                        raw_files.append(raw_file)
                    except Exception as e:
                        logger.error(f"Error fetching content for {path}: {e}")
                        continue
                        
        logger.info(f"Successfully fetched {len(raw_files)} files from {repo_name}")
        return raw_files

    def fetch_open_issues(self, repo_name: str) -> List[Dict[str, Any]]:
        """Fetch open issues from the repository."""
        repo = self.get_repo(repo_name)
        issues = []
        try:
            # Fetch only actual issues, not PRs (GitHub API returns PRs as issues too, but we can filter or keep)
            for issue in repo.get_issues(state="open"):
                if issue.pull_request is None: # It's a pure issue
                    issues.append({
                        "number": issue.number,
                        "title": issue.title,
                        "body": issue.body or "",
                        "url": issue.html_url,
                        "created_at": issue.created_at.isoformat() if issue.created_at else None,
                        "author": issue.user.login if issue.user else None
                    })
        except Exception as e:
            logger.error(f"Error fetching issues for {repo_name}: {e}")
        return issues

    def post_issue_comment(self, repo_name: str, issue_number: int, comment: str) -> bool:
        """Post a comment to a GitHub issue."""
        if not os.getenv("GITHUB_POST_COMMENTS", "false").lower() == "true":
            logger.info(f"Skipping comment on issue #{issue_number} (GITHUB_POST_COMMENTS is false)")
            return False
            
        try:
            repo = self.get_repo(repo_name)
            issue = repo.get_issue(number=issue_number)
            issue.create_comment(comment)
            logger.info(f"Successfully posted comment to issue #{issue_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to post comment to issue #{issue_number}: {e}")
            return False
