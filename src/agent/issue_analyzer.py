import os
import logging
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.schemas import IssueFix, CodeChunk
from src.retrieval.vector_store import CodeVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.ingestion.github_client import GithubIngestionClient
from src.agent.prompts import ISSUE_FIX_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class IssueAnalyzer:
    def __init__(
        self,
        vector_store: CodeVectorStore,
        reranker: CrossEncoderReranker,
        github_client: GithubIngestionClient
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.github_client = github_client
        
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        
        self.llm = ChatGroq(
            temperature=0.2,
            model_name=model_name,
            groq_api_key=api_key or "gsk_dummy_temp_testing_key"
        )
        
        # Bind structured output
        self.structured_llm = self.llm.with_structured_output(IssueFix)

    def analyze_and_suggest_fix(self, repo_name: str, issue_number: int) -> IssueFix:
        """
        Analyze a GitHub issue, retrieve relevant code, and generate a structured IssueFix suggestion.
        """
        logger.info(f"Analyzing issue #{issue_number} for repo '{repo_name}'")
        
        # 1. Fetch the issue details from GitHub
        repo = self.github_client.get_repo(repo_name)
        issue = repo.get_issue(number=issue_number)
        
        issue_title = issue.title
        issue_body = issue.body or ""
        
        logger.info(f"Retrieved Issue #{issue_number}: '{issue_title}'")
        
        # 2. Formulate search queries to find the code
        # We search with the title and description keywords
        search_query = f"{issue_title} {issue_body[:300]}"
        candidates = self.vector_store.search(repo_name, search_query, top_k=20)
        
        # Rerank to top 5
        relevant_chunks = self.reranker.rerank(search_query, candidates, top_n=5)
        
        # Format the context chunks for the prompt
        formatted_context = []
        for idx, chunk in enumerate(relevant_chunks):
            chunk_repr = (
                f"Chunk #{idx+1} | File: {chunk.file_path} (Lines: {chunk.start_line}-{chunk.end_line})\n"
                f"```{chunk.language}\n{chunk.content}\n```"
            )
            formatted_context.append(chunk_repr)
            
        context_str = "\n\n".join(formatted_context)
        
        # 3. Create the prompt for the structured LLM
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=ISSUE_FIX_SYSTEM_PROMPT),
            HumanMessage(content=f"""
Active Repository: {repo_name}
Issue Number: #{issue_number}
Issue Title: {issue_title}
Issue Description:
{issue_body}

Relevant Code Chunks retrieved from codebase:
{context_str}

Analyze the issue, propose a root cause hypothesis, determine the responsible file and function, and supply a suggested code change, explanation, and test case as specified in the structured schema.
""")
        ])
        
        # 4. Invoke LLM and get structured output
        logger.info("Requesting structured issue analysis from Groq LLM...")
        analysis_input = prompt.format_messages()
        issue_fix: IssueFix = self.structured_llm.invoke(analysis_input)
        
        logger.info(f"Analysis complete. Confident score: {issue_fix.confidence_score}")
        
        # 5. Stretch: Post a comment back on the GitHub issue if enabled
        if os.getenv("GITHUB_POST_COMMENTS", "false").lower() == "true":
            comment_md = self._format_as_github_comment(issue_fix)
            self.github_client.post_issue_comment(repo_name, issue_number, comment_md)
            
        return issue_fix

    def _format_as_github_comment(self, fix: IssueFix) -> str:
        """Format the IssueFix as a highly polished GitHub comment."""
        func_str = f" in function `{fix.responsible_function}`" if fix.responsible_function else ""
        
        comment = f"""### 🤖 Autonomous Code Assistant Analysis

I have analyzed this issue and searched the codebase. Here is my proposed analysis and fix:

#### 🔍 Root Cause Hypothesis
{fix.root_cause_hypothesis}

#### 🛠 Proposed Change
* **Responsible File:** `{fix.responsible_file}`{func_str}
* **Confidence:** `{fix.confidence_score * 100:.1f}%`

**Suggested Code Change:**
```diff
{fix.suggested_change}
```

**Explanation:**
{fix.explanation}

#### 🧪 Verification Test Case
{fix.test_case}

---
*Created automatically by the Autonomous Coding Assistant.*
"""
        return comment
