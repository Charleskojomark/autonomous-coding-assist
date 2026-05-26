import logging
import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.models.schemas import RawFile, CodeChunk

logger = logging.getLogger(__name__)

# Map our language identifier to LangChain Language enum
LANG_TO_LANGCHAIN = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "markdown": Language.MARKDOWN,
    "html": Language.HTML,
    "go": Language.GO,
    "rust": Language.RUST,
    "cpp": Language.CPP,
}

class CodeChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        lc_lang = LANG_TO_LANGCHAIN.get(language.lower())
        if lc_lang:
            return RecursiveCharacterTextSplitter.from_language(
                language=lc_lang,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

    def _find_exact_occurrences(self, full_text: str, chunk_text: str, search_offset: int) -> int:
        """Find the start index of the chunk_text in full_text starting from search_offset."""
        # Clean whitespaces in case splitter formatted/stripped them
        # Let's try exact match first
        idx = full_text.find(chunk_text, search_offset)
        if idx != -1:
            return idx
            
        # Fallback: line-based or fuzzy match if exact match fails
        # Splitting chunk_text into lines and trying to match the first few lines
        lines = [line.strip() for line in chunk_text.split('\n') if line.strip()]
        if not lines:
            return search_offset
            
        first_line = lines[0]
        # Find first line
        start_search = search_offset
        while True:
            idx = full_text.find(first_line, start_search)
            if idx == -1:
                break
            # Verify if it's a good match by checking subsequent characters
            return idx
            
        return search_offset

    def _extract_function_or_class(
        self, 
        lines: List[str], 
        start_line_idx: int, 
        end_line_idx: int,
        language: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Identify the class and function associated with the chunk.
        First scans inside the chunk lines [start_line_idx, end_line_idx].
        If not found, scans upwards from start_line_idx to find the enclosing scope.
        """
        class_name = None
        func_name = None
        
        # Python patterns
        python_class_pattern = re.compile(r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        python_func_pattern = re.compile(r'^\s*(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        
        # JS/TS patterns
        js_class_pattern = re.compile(r'(?:export\s+)?(?:default\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        js_func_patterns = [
            re.compile(r'(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)'),
            re.compile(r'(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'),
            re.compile(r'^\s*(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{') # method
        ]

        # 1. Search INSIDE the chunk first (downward scan)
        for i in range(start_line_idx, min(end_line_idx + 1, len(lines))):
            line = lines[i]
            if not line.strip():
                continue
                
            if language.lower() == "python":
                class_match = python_class_pattern.match(line)
                if class_match and not class_name:
                    class_name = class_match.group(1)
                func_match = python_func_pattern.match(line)
                if func_match and not func_name:
                    func_name = func_match.group(1)
                    
            elif language.lower() in ("javascript", "typescript"):
                class_match = js_class_pattern.search(line)
                if class_match and not class_name:
                    class_name = class_match.group(1)
                for pattern in js_func_patterns:
                    func_match = pattern.search(line)
                    if func_match and not func_name:
                        func_name = func_match.group(1)
                        break

        # 2. If function or class is not found inside, scan UPWARDS from start_line_idx
        if language.lower() == "python":
            if not class_name or not func_name:
                current_indent = None
                for i in range(min(start_line_idx, len(lines) - 1), -1, -1):
                    line = lines[i]
                    if not line.strip():
                        continue
                    
                    leading_spaces = len(line) - len(line.lstrip())
                    if current_indent is None:
                        current_indent = leading_spaces
                        
                    # Enclosing class
                    if not class_name:
                        class_match = python_class_pattern.match(line)
                        if class_match:
                            class_name = class_match.group(1)
                            
                    # Enclosing function
                    if not func_name:
                        func_match = python_func_pattern.match(line)
                        if func_match:
                            func_name = func_match.group(1)
                            # Keep scanning for class enclosing the function
                            for j in range(i - 1, -1, -1):
                                class_line = lines[j]
                                class_match = python_class_pattern.match(class_line)
                                if class_match:
                                    class_name = class_match.group(1)
                                    break
                            break
                            
        elif language.lower() in ("javascript", "typescript"):
            if not class_name or not func_name:
                for i in range(min(start_line_idx, len(lines) - 1), -1, -1):
                    line = lines[i]
                    if not line.strip():
                        continue
                        
                    # Enclosing class
                    if not class_name:
                        class_match = js_class_pattern.search(line)
                        if class_match:
                            class_name = class_match.group(1)
                            
                    # Enclosing function
                    if not func_name:
                        for pattern in js_func_patterns:
                            func_match = pattern.search(line)
                            if func_match:
                                func_name = func_match.group(1)
                                # Keep scanning for class
                                for j in range(i - 1, -1, -1):
                                    class_line = lines[j]
                                    class_match = js_class_pattern.search(class_line)
                                    if class_match:
                                        class_name = class_match.group(1)
                                        break
                                break
                        if func_name:
                            break

        return func_name, class_name

    def chunk_file(self, raw_file: RawFile, repo_name: str) -> List[CodeChunk]:
        """Split raw file content into chunks with rich metadata."""
        splitter = self._get_splitter(raw_file.language)
        
        # Use split_text to get chunk contents
        try:
            chunks_text = splitter.split_text(raw_file.content)
        except Exception as e:
            logger.error(f"Error splitting text for {raw_file.path}: {e}")
            # Fallback split
            chunks_text = [raw_file.content]
            
        file_lines = raw_file.content.split('\n')
        code_chunks = []
        search_offset = 0
        
        for idx, chunk_text in enumerate(chunks_text):
            if not chunk_text.strip():
                continue
                
            # Find the character index of the chunk inside the original content
            start_char_idx = self._find_exact_occurrences(raw_file.content, chunk_text, search_offset)
            end_char_idx = start_char_idx + len(chunk_text)
            
            # Update search offset for next iteration to support duplicate contents
            search_offset = start_char_idx + 1
            
            # Map characters to line numbers (1-based index)
            start_line = raw_file.content[:start_char_idx].count('\n') + 1
            end_line = raw_file.content[:end_char_idx].count('\n') + 1
            
            # Scan upwards from start_line to identify function or class
            func_name, class_name = self._extract_function_or_class(
                file_lines, 
                start_line - 1, 
                end_line - 1,
                raw_file.language
            )
            
            code_chunk = CodeChunk(
                file_path=raw_file.path,
                repo_name=repo_name,
                content=chunk_text,
                language=raw_file.language,
                function_name=func_name,
                class_name=class_name,
                chunk_index=idx,
                start_line=start_line,
                end_line=end_line
            )
            code_chunks.append(code_chunk)
            
        return code_chunks
