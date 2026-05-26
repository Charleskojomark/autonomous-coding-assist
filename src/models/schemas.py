from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class RawFile(BaseModel):
    path: str
    content: str
    size: int
    last_modified: Optional[datetime] = None
    language: str

class CodeChunk(BaseModel):
    file_path: str
    repo_name: str
    content: str
    language: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    chunk_index: int
    start_line: int
    end_line: int

class IssueFix(BaseModel):
    root_cause_hypothesis: str = Field(description="Hypothesis of why the issue is happening")
    responsible_file: str = Field(description="The specific file that needs to be modified")
    responsible_function: Optional[str] = Field(None, description="The specific function that needs to be modified")
    suggested_change: str = Field(description="The proposed code changes (e.g. in unified diff format or clear code blocks)")
    explanation: str = Field(description="Explanation of why this change fixes the issue")
    test_case: str = Field(description="A test case or steps to verify the fix")
    confidence_score: float = Field(description="Confidence score from 0.0 to 1.0")

class Citation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    citations: List[Citation] = []
