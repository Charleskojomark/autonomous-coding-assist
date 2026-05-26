import pytest
from src.models.schemas import RawFile
from src.ingestion.chunker import CodeChunker

def test_code_chunker_python_lines():
    chunker = CodeChunker(chunk_size=100, chunk_overlap=10)
    
    python_code = """class MyClass:
    def method_one(self):
        print("Hello World")
        
    def method_two(self):
        return 42
"""
    raw_file = RawFile(
        path="src/main.py",
        content=python_code,
        size=len(python_code),
        language="python"
    )
    
    chunks = chunker.chunk_file(raw_file, "owner/test-repo")
    
    assert len(chunks) > 0
    # Let's check first chunk metadata
    first_chunk = chunks[0]
    assert first_chunk.language == "python"
    assert first_chunk.repo_name == "owner/test-repo"
    assert first_chunk.file_path == "src/main.py"
    assert first_chunk.start_line == 1
    assert first_chunk.class_name == "MyClass"

def test_code_chunker_javascript_function():
    chunker = CodeChunker(chunk_size=100, chunk_overlap=10)
    
    js_code = """export class AuthController {
    async login(req, res) {
        const token = jwt.sign({ id: req.user.id });
        return res.json({ token });
    }
}
"""
    raw_file = RawFile(
        path="src/auth.js",
        content=js_code,
        size=len(js_code),
        language="javascript"
    )
    
    chunks = chunker.chunk_file(raw_file, "owner/test-repo")
    assert len(chunks) > 0
    assert chunks[0].class_name == "AuthController"
    assert chunks[0].function_name == "login"
