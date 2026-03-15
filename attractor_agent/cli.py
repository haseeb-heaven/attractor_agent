import os
import re
import sys
import socket
import threading
import logging
import subprocess
import time
import atexit
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.status import Status
from rich.logging import RichHandler
from rich.tree import Tree

from attractor.pipeline.engine import PipelineConfig, run_pipeline
from attractor.pipeline.events import EventEmitter, PipelineEventKind, PipelineEvent
from attractor.pipeline.interviewer import Interviewer, Question, Answer
from attractor.pipeline.backend import LLMBackend

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler("attractor_agent.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("attractor_agent.cli")

console = Console()


@dataclass
class ExtractedBlock:
    """Structured extracted code block."""
    language: str
    code: str
    filename_comment: str | None = None
    header_filename: str | None = None
    attribute_filename: str | None = None


# ── Port Waiter ───────────────────────────────────────────────────────────────
def wait_for_port(host: str, port: int, timeout: float = 45.0) -> bool:
    """
    Wait until a TCP listener at the given host and port accepts a connection or raise on timeout.
    
    Parameters:
        timeout (float): Maximum time to wait in seconds.
    
    Returns:
        True if a connection to the host:port was accepted before the timeout.
    
    Raises:
        RuntimeError: If the host:port did not accept a connection within the specified timeout.
    """
    logger.info(f"Waiting for server at {host}:{port} (timeout={timeout}s)")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                logger.info(f"✓ Server ready at {host}:{port}")
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise RuntimeError(
        f"Server at {host}:{port} did not respond within {timeout}s — "
        "is Node.js installed? Run: node --version"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', text.lower())
    return re.sub(r'_+', '_', slug).strip('_')


def get_extension(language: str) -> str:
    """Map language name to primary file extension."""
    lang = language.lower()
    mapping = {
        "javascript": ".js",
        "js":         ".js",
        "typescript": ".ts",
        "ts":         ".ts",
        "html":       ".html",
        "html/css":   ".html",
        "go":         ".go",
        "rust":       ".rs",
        "c++":        ".cpp",
        "cpp":        ".cpp",
        "java":       ".java",
    }
    for key, ext in mapping.items():
        if key in lang:
            return ext
    return ".py"


def get_file_extension_from_tag(lang_tag: str, fallback_language: str) -> str:
    """
    Determine a filename extension corresponding to a code-fence language tag.
    
    Parameters:
        lang_tag (str): Language identifier from a fenced code block (case and surrounding whitespace are ignored).
        fallback_language (str): Language name to use if the tag is not recognized.
    
    Returns:
        str: File extension including the leading dot (e.g., ".py", ".js").
    """
    ext_map = {
        "javascript": ".js",
        "js":         ".js",
        "typescript": ".ts",
        "ts":         ".ts",
        "html":       ".html",
        "css":        ".css",
        "sql":        ".sql",
        "python":     ".py",
        "go":         ".go",
        "rust":       ".rs",
        "java":       ".java",
        "cpp":        ".cpp",
        "c++":        ".cpp",
        "csharp":     ".cs",
        "c#":         ".cs",
        "bash":       ".sh",
        "shell":      ".sh",
        "json":       ".json",
        "yaml":       ".yml",
        "yml":        ".yml",
        "markdown":   ".md",
        "md":         ".md",
        "xml":        ".xml",
        "toml":       ".toml",
    }
    return ext_map.get(lang_tag.lower().strip(), get_extension(fallback_language))


def get_smart_filename(lang_tag: str, index: int, language: str) -> str:
    """
    Choose a descriptive filename for a code block based on its language tag and occurrence index.
    
    Parameters:
        lang_tag (str): Language tag from a code fence or other hint (e.g., "python", "js", "c#").
        index (int): Zero-based occurrence index used to select a name from a curated list or to generate a numbered fallback.
        language (str): Fallback language name used to determine the file extension when `lang_tag` does not map directly.
    
    Returns:
        filename (str): A chosen filename for the block (a curated name for the detected extension, or `file_N.<ext>` when the curated list is exhausted).
    """
    ext = get_file_extension_from_tag(lang_tag, language)
    name_map: dict[str, list[str]] = {
        ".html": ["index.html", "login.html", "register.html", "dashboard.html",
                  "profile.html", "404.html"],
        ".css":  ["style.css", "auth.css", "dashboard.css", "components.css"],
        ".js":   ["main.js", "app.js", "api.js", "utils.js", "db.js", "routes.js"],
        ".ts":   ["main.ts", "app.ts", "api.ts", "utils.ts", "db.ts"],
        ".sql":  ["schema.sql", "seed.sql", "migrations.sql"],
        ".py":   ["main.py", "app.py", "models.py", "utils.py", "routes.py",
                  "database.py", "config.py"],
        ".sh":   ["setup.sh", "run.sh", "build.sh"],
        ".json": ["package.json", "config.json", "tsconfig.json"],
        ".java": ["Main.java", "App.java", "Utils.java", "Database.java"],
        ".cpp":  ["main.cpp", "app.cpp", "utils.cpp"],
        ".go":   ["main.go", "app.go", "utils.go"],
    }
    names = name_map.get(ext, [])
    if index < len(names):
        return names[index]
    return f"file_{index + 1}{ext}"


def _extract_filename_comment(text: str) -> str | None:
    """
    Extract a filename annotation from the first up-to-five lines of a text block.
    
    Scans the beginning of `text` for a filename marker using common comment styles (e.g., `// filename:`, `# filename:`, `/* filename: */`, `<!-- filename: -->`, or a bare `filename:`) and returns the annotated filename when present.
    
    Parameters:
        text (str): Text to scan for a filename annotation.
    
    Returns:
        str | None: The extracted filename string if found, otherwise `None`.
    """
    patterns = [
        r"^\s*//\s*filename\s*:\s*(.+?)\s*$",
        r"^\s*#\s*filename\s*:\s*(.+?)\s*$",
        r"^\s*/\*\s*filename\s*:\s*(.+?)\s*\*/\s*$",
        r"^\s*<!--\s*filename\s*:\s*(.+?)\s*-->\s*$",
        r"^\s*filename\s*:\s*(.+?)\s*$",
    ]
    for line in text.splitlines()[:5]:
        for pattern in patterns:
            m = re.match(pattern, line, re.IGNORECASE)
            if m:
                return m.group(1).strip()
    return None


def _strip_filename_markers(text: str) -> str:
    """
    Remove filename-annotation lines from a block of text and return the cleaned text.
    
    This strips lines that contain filename annotations (for example lines starting with common comment-based filename markers such as "// filename:", "# filename:", "/* filename:", "<!-- filename:", or similar) and returns the remaining lines joined by newlines with surrounding whitespace trimmed.
    
    Parameters:
        text (str): The input text potentially containing filename-annotation lines.
    
    Returns:
        str: The input text with filename-annotation lines removed, joined by newlines, and trimmed of leading and trailing whitespace.
    """
    out: list[str] = []
    for line in text.splitlines():
        if _extract_filename_comment(line):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _parse_info_attributes(info: str) -> tuple[str, str | None]:
    """
    Parse a fenced-code info string into its language token and an optional filename attribute.
    
    Parameters:
        info (str): The info string from a Markdown fenced code block (text after the opening backticks), e.g. "python filename=app.py" or "js file='index.js'".
    
    Returns:
        tuple[str, str | None]: A tuple where the first element is the language token (empty string if absent) and the second element is the extracted filename if present, otherwise `None`.
    """
    info = info.strip()
    if not info:
        return "", None
    parts = info.split()
    lang = parts[0]
    attr_filename = None
    for pattern in [r"(?:file|filename|name)\s*=\s*\"([^\"]+)\"", r"(?:file|filename|name)\s*=\s*'([^']+)'", r"(?:file|filename|name)\s*=\s*([^\s]+)"]:
        m = re.search(pattern, info, re.IGNORECASE)
        if m:
            attr_filename = m.group(1).strip()
            break
    return lang, attr_filename


def _extract_markdown_blocks(text: str) -> list[ExtractedBlock]:
    """
    Extract code blocks from Markdown fenced code sections into structured ExtractedBlock entries.
    
    Parses triple-backtick fenced blocks, skips empty blocks, and for each block:
    - determines the language and optional attribute filename from the fence info string,
    - extracts an inline filename comment if present,
    - strips filename marker lines from the block content.
    
    Parameters:
        text (str): Markdown text to scan for fenced code blocks.
    
    Returns:
        list[ExtractedBlock]: A list of ExtractedBlock objects for each non-empty fenced code block. Each entry has `language` lowercased, `code` with filename markers removed, and `filename_comment` / `attribute_filename` populated when detected.
    """
    blocks: list[ExtractedBlock] = []
    for info, code in re.findall(r"```([^\n]*)\n(.*?)```", text, re.DOTALL):
        cleaned = code.strip()
        if not cleaned:
            continue
        lang, attr_filename = _parse_info_attributes(info)
        filename_comment = _extract_filename_comment(cleaned)
        blocks.append(
            ExtractedBlock(
                language=lang.lower().strip(),
                code=_strip_filename_markers(cleaned),
                filename_comment=filename_comment,
                attribute_filename=attr_filename,
            )
        )
    return blocks


def _extract_fallback_marker_blocks(text: str) -> list[ExtractedBlock]:
    """
    Extract code blocks delimited by informal start/end markers (e.g., "[start code]", "<start file>", "begin snippet") and return them as ExtractedBlock objects.
    
    Scans the given text for loose start/end markers used when fenced Markdown blocks are not present, captures the intervening lines as code, and preserves any associated language or filename attributes when available.
    
    Parameters:
        text (str): The source text to scan for fallback marker delimited code blocks.
    
    Returns:
        list[ExtractedBlock]: A list of extracted code blocks. Each ExtractedBlock contains:
            - language: the parsed language tag if present (empty string if not detected),
            - code: the block contents with filename marker lines removed,
            - filename_comment: a filename found inside the block as a comment (if any),
            - attribute_filename: a filename provided inline with the start marker (if any).
    """
    start_re = re.compile(
        r"^\s*(?:\[|<)?\s*(?:start|begin)?\s*(?:code|file|block|snippet)\b(.*?)(?:\]|>)?\s*$",
        re.IGNORECASE,
    )
    end_re = re.compile(
        r"^\s*(?:\[|<)?\s*(?:end|stop|close)\s*(?:code|file|block|snippet)?\b.*(?:\]|>)?\s*$",
        re.IGNORECASE,
    )

    blocks: list[ExtractedBlock] = []
    current_lines: list[str] = []
    current_lang = ""
    current_attr_filename: str | None = None
    inside_block = False

    def flush() -> None:
        """
        Convert the accumulated current_lines into an ExtractedBlock and append it to blocks when there is non-empty content.
        
        If current_lines is empty or contains only whitespace, this function does nothing. Otherwise it joins the lines, trims surrounding whitespace, removes in-band filename markers from the code, extracts any filename comment found within the code, constructs an ExtractedBlock using the current language and attribute filename, and appends it to the blocks list.
        """
        if not current_lines:
            return
        code = "\n".join(current_lines).strip()
        if not code:
            return
        blocks.append(
            ExtractedBlock(
                language=current_lang,
                code=_strip_filename_markers(code),
                filename_comment=_extract_filename_comment(code),
                attribute_filename=current_attr_filename,
            )
        )

    for line in text.splitlines():
        start_match = start_re.match(line)
        if start_match:
            if inside_block:
                flush()
            current_lines = []
            inside_block = True
            attr = start_match.group(1).strip()
            current_lang, current_attr_filename = _parse_info_attributes(attr)
            continue
        if inside_block and end_re.match(line):
            flush()
            current_lines = []
            current_lang = ""
            current_attr_filename = None
            inside_block = False
            continue
        if inside_block:
            current_lines.append(line)

    if inside_block:
        flush()
    return [b for b in blocks if b.code]


def _extract_header_sections(text: str) -> list[ExtractedBlock]:
    """
    Extract code blocks that are introduced by header-style lines and record the header as the block's filename.
    
    This scans `text` for header lines of the forms:
    - === Title ===
    - --- Title ---
    - ### Title
    - FILE: name
    Headers are matched case-insensitively. For each header found, the subsequent contiguous lines (up to the next header) are collected as the block's code and stored in an ExtractedBlock with `header_filename` set to the header text and `language` left empty.
    Returns:
        list[ExtractedBlock]: Extracted blocks where `code` is the trimmed text following each header and `header_filename` is the header title. Empty or whitespace-only blocks are omitted.
    """
    header_re = re.compile(
        r"^\s*(?:===\s*(.+?)\s*===|---\s*(.+?)\s*---|###\s+(.+?)\s*|FILE\s*:\s*(.+?)\s*)$",
        re.IGNORECASE,
    )
    blocks: list[ExtractedBlock] = []
    current_filename: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            if current_filename and current_lines:
                code = "\n".join(current_lines).strip()
                if code:
                    blocks.append(ExtractedBlock(language="", code=code, header_filename=current_filename))
            current_filename = next((g.strip() for g in m.groups() if g), None)
            current_lines = []
            continue
        if current_filename:
            current_lines.append(line)

    if current_filename and current_lines:
        code = "\n".join(current_lines).strip()
        if code:
            blocks.append(ExtractedBlock(language="", code=code, header_filename=current_filename))
    return blocks


def _extract_filename_comment_sections(text: str) -> list[ExtractedBlock]:
    """
    Extract code sections that are preceded by filename-style comment markers and return them as ExtractedBlock items.
    
    Scans the input text for lines that contain filename comment annotations (e.g., comments that indicate a filename) and collects subsequent lines as the code body for that filename until the next filename comment or end of text.
    
    Parameters:
        text (str): Multiline string to scan for filename-comment-marked sections.
    
    Returns:
        list[ExtractedBlock]: A list of ExtractedBlock instances where each block's filename_comment is set to the discovered filename and code contains the trimmed lines that followed that filename marker. Blocks with no code are omitted.
    """
    blocks: list[ExtractedBlock] = []
    current_filename: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        marker = _extract_filename_comment(line)
        if marker:
            if current_filename and current_lines:
                blocks.append(
                    ExtractedBlock(language="", code="\n".join(current_lines).strip(), filename_comment=current_filename)
                )
            current_filename = marker
            current_lines = []
            continue
        if current_filename:
            current_lines.append(line)

    if current_filename and current_lines:
        blocks.append(ExtractedBlock(language="", code="\n".join(current_lines).strip(), filename_comment=current_filename))
    return [b for b in blocks if b.code]


def extract_blocks_with_fallbacks(text: str) -> list[ExtractedBlock]:
    """
    Attempt to extract structured code blocks from the provided text using multiple parsing strategies in priority order.
    
    Parameters:
        text (str): The input text to scan for code blocks and filename metadata.
    
    Returns:
        list[ExtractedBlock]: A list of ExtractedBlock objects found in the input. Parsing strategies are attempted in this order: Markdown fenced code blocks, start/end marker blocks, header-delimited sections, and filename-comment-delimited sections; the first strategy that yields any blocks is used.
    """
    markdown_blocks = _extract_markdown_blocks(text)
    if markdown_blocks:
        return markdown_blocks

    marker_blocks = _extract_fallback_marker_blocks(text)
    if marker_blocks:
        return marker_blocks

    header_blocks = _extract_header_sections(text)
    if header_blocks:
        return header_blocks

    return _extract_filename_comment_sections(text)


def print_project_tree(project_dir: Path) -> None:
    """Print a Rich directory tree of all generated project files."""
    tree = Tree(f"[bold green]📁 {project_dir}[/bold green]")
    for f in sorted(project_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            icon = "📄"
            if f.suffix in (".html", ".css"):
                icon = "🌐"
            elif f.suffix in (".js", ".ts"):
                icon = "⚡"
            elif f.suffix == ".py":
                icon = "🐍"
            elif f.suffix == ".sql":
                icon = "🗄️"
            elif f.suffix == ".md":
                icon = "📝"
            elif f.suffix in (".sh",):
                icon = "🔧"
            tree.add(
                f"{icon} [cyan]{f.name}[/cyan] [dim]({size:,} bytes)[/dim]"
            )
    console.print(tree)


# ── DOT Builder ───────────────────────────────────────────────────────────────
def build_dot(request: str, language: str, framework: str,
              include_tests: bool, include_sdlc: bool) -> str:
    """
    Build a spec-correct Attractor DOT pipeline with:
      Plan → Generate → [Tests → TestGate] → Review → [SDLCCheck] → Done
    """
    fw = f"using {framework}" if framework else ""
    logger.debug(
        f"Building DOT — lang={language} fw='{framework}' "
        f"tests={include_tests} sdlc={include_sdlc}"
    )

    dot = f'''digraph generate_app {{
    rankdir=LR;
    goal="Build: {request} in {language} {fw}";

    node [shape=box, timeout="1200s", max_retries=2]

    Start [shape=Mdiamond, label="Start"];
    Done  [shape=Msquare,  label="Done"];

    Plan [
        label="Plan Architecture",
        prompt="Plan architecture for: {request} in {language} {fw}.\\nOutput:\\n- Full folder and file structure with exact filenames\\n- Each module/file purpose\\n- All dependencies list\\n- Database schema if needed\\nNO CODE YET.",
        goal_gate=true
    ];

    Generate [
        label="Generate Code",
        prompt="Using the plan above, write COMPLETE {language} code for: {request} {fw}.\\nStrict rules:\\n- Output EVERY file as a SEPARATE triple backtick code block\\n- Label EVERY block with its language tag e.g. ```html ```css ```javascript ```sql ```python\\n- First line inside EVERY block must be a comment: // filename: <exact-filename> (or # filename: for Python/SQL)\\n- ALL files must be complete, functional and runnable\\n- No explanations or text between blocks",
        goal_gate=true
    ];
'''

    if include_tests:
        dot += f'''
    Tests [
        label="Unit Tests",
        prompt="Write COMPLETE unit tests for ALL functions in the code above using {language}.\\n- Cover all edge cases\\n- Use the standard test framework for {language}\\n- Output ONLY test code in a single labeled triple backtick block\\n- First line: // filename: test_main{get_extension(language)}",
        goal_gate=true
    ];

    RunTests [handler="test_runner", label="Execute Tests", max_retries=2];
'''

    if include_sdlc:
        dot += '''
    SDLCCheck [
        label="SDLC Validation",
        prompt="Review ALL generated code files for:\\n1. Error handling and exceptions\\n2. Input validation and sanitization\\n3. Security (SQL injection, XSS, auth checks)\\n4. Logging and observability\\n5. Code documentation\\n6. Memory and performance issues\\nOutput: PASS if everything is good, or a numbered LIST of specific issues per file.",
        goal_gate=true
    ];
'''

    dot += '''
    Score [handler="satisfaction_scorer", label="Satisfaction Scorer"];
    DeployTwin [handler="digital_twin", label="Deploy to Digital Twin Universe"];
    Review [shape=hexagon, label="Human Review"];

    Start -> Plan -> Generate;
'''

    if include_tests:
        dot += '''    Generate -> Tests -> RunTests;
    RunTests -> Score [label="pass", condition="outcome=success"];
    RunTests -> Generate [label="retry_code", condition="outcome=fail"];
'''
    else:
        dot += '''    Generate -> Score;
'''

    if include_sdlc:
        dot += '''    Score -> SDLCCheck;
    SDLCCheck -> Review [condition="outcome=success"];
    SDLCCheck -> Generate [condition="outcome=fail", label="[F]ix"];
'''
    else:
        dot += '''    Score -> Review;
'''

    dot += '''    Review -> DeployTwin [label="[A]pprove"];
    DeployTwin -> Done;
    Review -> Generate [label="[R]etry"];
}
'''
    return dot


# ── File Saver ────────────────────────────────────────────────────────────────
def _infer_language_from_filename(filename: str) -> str:
    """
    Infer a programming language identifier from a filename's extension.
    
    Returns:
        language (str): Lowercase language name such as "python" or "javascript" if the extension is recognized, otherwise an empty string.
    """
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".sh": "bash",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cpp": "cpp",
        ".cs": "csharp",
    }
    return mapping.get(suffix, "")


def _default_filename_for_language(lang: str, index: int) -> str:
    """
    Produce a conventional default filename for a given programming language and a zero-based index.
    
    If the language has a well-known primary filename (e.g., "main.py", "Main.java") the function returns that name for index 0 and returns the same name with an appended ordinal suffix before the extension for higher indices (e.g., "main_2.py"). If the language is not recognized, the function returns a generic name of the form "file_N<ext>" where N is index+1 and <ext> is inferred from the language tag or falls back to ".py".
    
    Parameters:
        lang (str): Language name or fenced-code tag (e.g., "python", "js", "csharp").
        index (int): Zero-based index used to disambiguate multiple files.
    
    Returns:
        str: A filesystem-safe filename appropriate for the language and index (for example "Main.java", "main_2.py", or "file_1.py").
    """
    base_map = {
        "java": "Main.java",
        "csharp": "Program.cs",
        "c#": "Program.cs",
        "go": "main.go",
        "cpp": "main.cpp",
        "c++": "main.cpp",
        "rust": "main.rs",
        "python": "main.py",
        "javascript": "app.js",
        "js": "app.js",
        "typescript": "main.ts",
        "ts": "main.ts",
        "html": "index.html",
        "css": "styles.css",
        "sql": "schema.sql",
    }
    fallback = base_map.get(lang.lower().strip())
    if fallback and index == 0:
        return fallback
    if fallback:
        stem = Path(fallback).stem
        suffix = Path(fallback).suffix
        return f"{stem}_{index + 1}{suffix}"
    ext = get_file_extension_from_tag(lang, "python")
    return f"file_{index + 1}{ext}"


def _resolve_filename(block: ExtractedBlock, index: int, language: str) -> str:
    """
    Determine the target filename for an extracted code block.
    
    Parameters:
        block (ExtractedBlock): The extracted block containing possible filename hints and a language tag.
        index (int): Index used to generate a default filename when no explicit name is found.
        language (str): Fallback language to use when inferring a default filename.
    
    Returns:
        filename (str): The chosen filename. Prefers explicit hints in this order: `filename_comment`, `header_filename`, `attribute_filename`. If none are present, infers a language (from the block or provided fallback) and returns a conventional default filename based on that language and the provided index.
    """
    if block.filename_comment:
        return block.filename_comment
    if block.header_filename:
        return block.header_filename
    if block.attribute_filename:
        return block.attribute_filename

    detected_lang = block.language or _infer_language_from_filename(block.filename_comment or "") or language
    return _default_filename_for_language(detected_lang, index)


def save_output_files(
    result,
    project_dir: Path,
    language: str,
    app_file_name: str
) -> list[Path]:
    """
    Save extracted code blocks from the pipeline result into files under the project directory.
    
    Uses multi-stage fallback parsing to extract code blocks from the pipeline's Generate output, writes each block to an appropriate filename (creating directories as needed), and persists a README.md summarizing the project. Also saves detected tests and stores an extracted files manifest at the context key "Generate.extracted_files".
    
    Parameters:
        result: Pipeline run result object that exposes a .context for retrieving outputs and storing the extracted manifest.
        project_dir (Path): Target directory where extracted files and README.md will be written.
        language (str): Primary language of the generated project; used to choose file extensions and run instructions when generating the README.
        app_file_name (str): Suggested main application filename (used as a hint when resolving filenames for extracted blocks).
    
    Returns:
        list[Path]: List of file paths that were successfully written to disk.
    """
    generate_output = (
        result.context.get_string("Generate.output", "")
        or result.context.get_string("generate_output", "")
        or result.context.get_string("last_response", "")
        or result.context.get_string("output", "")
    )
    tests_output = (
        result.context.get_string("Tests.output", "")
        or result.context.get_string("tests_output", "")
    )

    logger.debug("=== ALL CONTEXT KEYS ===")
    try:
        for key in result.context.keys():
            val = str(result.context.get_string(key, ""))[:200]
            logger.debug(f"  KEY [{key}] => {val!r}")
    except Exception as e:
        logger.warning(f"Could not iterate context keys: {e}")
    logger.debug("========================")

    if not generate_output:
        logger.error("Generate output is EMPTY — context key mismatch. Check attractor_agent.log.")
        console.print(
            "[bold red]❌ No generated code found in pipeline context.\n"
            "   Check attractor_agent.log for available KEY names.[/bold red]"
        )
        return []

    extracted_blocks = extract_blocks_with_fallbacks(generate_output)
    logger.info(f"Found {len(extracted_blocks)} extracted block(s) in Generate output")

    if not extracted_blocks:
        logger.error("No structured blocks detected in generated output.")
        logger.error("Raw output preview: %r", generate_output[:800])
        return []

    saved_files: list[Path] = []
    extracted_manifest: list[dict[str, str]] = []

    for idx, block in enumerate(extracted_blocks):
        code = block.code.strip()
        if not code:
            continue

        filename = _resolve_filename(block, idx, language)
        fpath = project_dir / filename

        try:
            os.makedirs(fpath.parent, exist_ok=True)
            fpath.write_text(code, encoding="utf-8")
            saved_files.append(fpath)
            extracted_manifest.append({
                "filename": filename,
                "language": block.language or _infer_language_from_filename(filename) or language.lower(),
                "content": code,
            })
            logger.info(f"Saved: {fpath} ({len(code):,} chars)")
            console.print(
                f"  [green]✓[/green] Saved [bold]{filename}[/bold] "
                f"[dim]({len(code):,} chars)[/dim]"
            )
        except OSError as e:
            logger.error(f"Failed to save {fpath}: {e}")
            console.print(f"  [red]✗[/red] Could not save {filename}: {e}")

    result.context.set("Generate.extracted_files", extracted_manifest)

    if tests_output:
        test_blocks = _extract_markdown_blocks(tests_output)
        test_code = test_blocks[0].code.strip() if test_blocks else tests_output.strip()
        test_ext = get_extension(language)
        test_file = project_dir / f"test_main{test_ext}"
        try:
            test_file.write_text(test_code, encoding="utf-8")
            saved_files.append(test_file)
            logger.info(f"Tests saved: {test_file} ({len(test_code):,} chars)")
            console.print(
                f"  [green]✓[/green] Tests saved [bold]{test_file.name}[/bold] "
                f"[dim]({len(test_code):,} chars)[/dim]"
            )
        except OSError as e:
            logger.error(f"Failed to save tests: {e}")

    run_instructions = {
        "python":     "```bash\npip install -r requirements.txt\npython main.py\n```",
        "javascript": "```bash\nnpm install\nnode main.js\n```",
        "typescript": "```bash\nnpm install\nnpx ts-node main.ts\n```",
        "java":       "```bash\njavac Main.java\njava Main\n```",
        "c++":        "```bash\ng++ -std=c++17 -o app main.cpp\n./app\n```",
        "go":         "```bash\ngo run main.go\n```",
        "rust":       "```bash\ncargo run\n```",
        "html":       "Open `index.html` in your browser.",
    }
    run_cmd = run_instructions.get(language.lower(), "See language documentation.")

    readme_lines = [
        "# Generated Project\n\n",
        "**Built by:** Attractor Agent 🚀\n",
        f"**Language:** {language}\n\n",
        "## Project Files\n\n",
    ]
    for f in saved_files:
        readme_lines.append(f"- `{f.name}`\n")

    readme_lines.append(f"\n## How to Run\n\n{run_cmd}\n")

    readme_file = project_dir / "README.md"
    try:
        readme_file.write_text("".join(readme_lines), encoding="utf-8")
        logger.info(f"README saved: {readme_file}")
        console.print("  [green]✓[/green] README saved [bold]README.md[/bold]")
    except OSError as e:
        logger.error(f"Failed to save README: {e}")

    return saved_files


# ── Rich Interviewer ──────────────────────────────────────────────────────────
class RichInterviewer(Interviewer):
    """Handles hexagon human-review gate prompts in the Rich TUI."""

    def __init__(self, console_lock: threading.Lock, status: Status):
        self.console_lock = console_lock
        self.status = status

    def ask(self, question: Question) -> Answer:
        logger.info(f"Human review gate triggered: node={question.node_id}")
        self.status.stop()
        with self.console_lock:
            console.print("\n")
            console.print(Panel(
                Markdown(question.text),
                title=f"[bold yellow]👁 Human Review[/bold yellow] "
                      f"[dim]({question.node_id})[/dim]",
                border_style="yellow",
                padding=(1, 2)
            ))
            options_text = "  ".join(
                f"[[bold]{option.key.upper()}[/bold]] {option.label}"
                for option in question.options
            )
            console.print(f"\n[bold cyan]Options:[/bold cyan]  {options_text}\n")
            valid_keys = [option.key.lower() for option in question.options]

            while True:
                choice = Prompt.ask("[bold green]Your choice[/bold green]").strip().lower()
                for opt in question.options:
                    if (
                        choice == opt.key.lower()
                        or choice == opt.label.lower()
                        or (choice and opt.label.lower().startswith(choice))
                    ):
                        logger.info(f"User selected: '{opt.label}'")
                        self.status.start()
                        return Answer(
                            question_id=question.id,
                            selected_label=opt.label
                        )
                console.print(
                    f"[red]❌ Invalid choice.[/red] "
                    f"Valid keys: [bold]{', '.join(valid_keys)}[/bold]"
                )


# ── Main CLI ──────────────────────────────────────────────────────────────────
def run_cli() -> None:
    """
    Run the interactive Attractor Agent command-line workflow.
    
    Prompts the user for a build request, language, framework and options, constructs a pipeline specification,
    optionally launches a mock LLM, executes the pipeline, and persists generated outputs to a project
    directory under projects/<slug>. Side effects include creating the project directory and files
    (e.g., pipeline.dot and generated source files), possibly starting a subprocess for a mock LLM,
    writing log entries to the configured logger, and exiting the process on fatal errors.
    """
    console.print(Panel(
        "Welcome to [bold magenta]Attractor Agent[/bold magenta] 🚀\n"
        "[dim]Full SDLC AI pipeline — Plan → Generate → Test → Review → Done[/dim]",
        expand=False,
        border_style="magenta",
        padding=(1, 4)
    ))
    logger.info("=" * 60)
    logger.info("Attractor Agent CLI started")
    logger.info("=" * 60)

    # ── Collect user inputs ───────────────────────────────────────────────────
    request = Prompt.ask("[bold cyan]What do you want to build?[/bold cyan]").strip()
    if not request:
        console.print("[red]No request provided. Exiting.[/red]")
        logger.error("Empty build request — exiting")
        sys.exit(1)

    language = Prompt.ask(
        "[bold cyan]Programming language?[/bold cyan]",
        choices=["Python", "JavaScript", "TypeScript", "HTML",
                 "Go", "Rust", "C++", "Java"],
        default="Python"
    )
    framework = Prompt.ask(
        "[bold cyan]Framework? (Enter to skip)[/bold cyan]",
        default=""
    ).strip()
    include_tests = Prompt.ask(
        "[bold cyan]Include unit tests? (y/n)[/bold cyan]",
        default="y"
    ).strip().lower() == "y"
    include_sdlc = Prompt.ask(
        "[bold cyan]Include SDLC review? (y/n)[/bold cyan]",
        default="y"
    ).strip().lower() == "y"
    use_mock = Prompt.ask(
        "[bold cyan]Use Mock LLM for testing? (localhost:5555) (y/n)[/bold cyan]",
        default="n"
    ).strip().lower() == "y"

    logger.info(
        f"Config — request='{request[:80]}' lang={language} "
        f"fw='{framework}' tests={include_tests} "
        f"sdlc={include_sdlc} mock={use_mock}"
    )

    # ── Project directory setup ───────────────────────────────────────────────
    slug = slugify(request)[:30] or "project"
    project_dir = Path("projects") / slug
    os.makedirs(project_dir, exist_ok=True)
    app_file_name = f"main{get_extension(language)}"
    logger.info(f"Project dir: {project_dir} | Main file: {app_file_name}")
    console.print(f"[dim]📁 Project folder: [bold]{project_dir}[/bold][/dim]")

    # ── Build and save DOT ────────────────────────────────────────────────────
    dot_content = build_dot(request, language, framework, include_tests, include_sdlc)
    dot_file = project_dir / "pipeline.dot"
    dot_file.write_text(dot_content, encoding="utf-8")
    console.print(f"[dim]🔧 Pipeline DOT saved: {dot_file}[/dim]")
    logger.info(f"DOT saved: {dot_file}")

    # ── Backend setup: Mock or Real ───────────────────────────────────────────
    if use_mock:
        console.print(
            "\n[bold yellow]🚀 Starting Mock LLM server on port 5555...[/bold yellow]"
        )
        logger.info("Launching mock LLM: npx -y @copilotkit/llmock --port 5555")

        # Use shell=True with a string command — fixes Windows npx.cmd issue
        mock_process = subprocess.Popen(
            "npx -y @copilotkit/llmock --port 5555",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Ensure mock server is killed when CLI exits
        atexit.register(lambda: mock_process.terminate())

        try:
            # Wait until the port is actually accepting connections
            wait_for_port("localhost", 5555, timeout=45)
            console.print("[dim]✓ Mock LLM server is ready.[/dim]\n")
        except RuntimeError as e:
            logger.error(f"Mock server startup failed: {e}")
            console.print(f"[bold red]❌ {e}[/bold red]")
            sys.exit(1)

        from attractor.llm.client import Client
        from attractor.llm.adapters.openai import OpenAIAdapter

        mock_adapter = OpenAIAdapter(
            api_key="mock-key",
            base_url="http://localhost:5555/v1"
        )
        client = Client(providers={"mock": mock_adapter}, default_provider="mock")
        backend = LLMBackend(client=client)
        logger.info("Mock LLM backend ready at http://localhost:5555/v1")

    else:
        backend = LLMBackend()
        logger.info("Using default LLMBackend (OpenRouter / configured provider)")

    # ── Pipeline setup ────────────────────────────────────────────────────────
    console_lock = threading.Lock()
    status = Status(
        "[bold green]Running pipeline...[/bold green]",
        console=console
    )
    interviewer = RichInterviewer(console_lock, status)

    config = PipelineConfig(
        simulate=False,
        codergen_backend=backend,
        interviewer=interviewer,
        checkpoint_dir=str(project_dir)
    )

    emitter = EventEmitter()

    # ── Event handler ─────────────────────────────────────────────────────────
    @emitter.on
    def on_event(event: PipelineEvent) -> None:
        msg = event.message or ""
        node = event.node_id
        if event.kind == PipelineEventKind.STAGE_STARTED:
            logger.info(f"[STAGE STARTED] {node}")
            with console_lock:
                status.update(f"[bold blue]⏳ Running:[/bold blue] {node}")
        elif event.kind == PipelineEventKind.STAGE_COMPLETED:
            logger.info(f"[STAGE COMPLETED] {node}")
            with console_lock:
                console.print(f"  [green]✓[/green] Completed: [bold]{node}[/bold]")
        elif event.kind == PipelineEventKind.STAGE_FAILED:
            logger.error(f"[STAGE FAILED] {node} — {msg}")
            with console_lock:
                console.print(
                    f"  [red]✗[/red] Failed: [bold]{node}[/bold] "
                    f"[dim]— {msg}[/dim]"
                )

    # ── Execute pipeline ──────────────────────────────────────────────────────
    console.print(
        f"\n[bold]Starting pipeline for:[/bold] [magenta]{request}[/magenta]\n"
    )
    status.start()
    try:
        logger.info("Pipeline execution started")
        result = run_pipeline(dot_content, config=config, emitter=emitter)
    except Exception as e:
        logger.exception(f"Pipeline crashed with unhandled exception: {e}")
        console.print(f"\n[bold red]❌ Pipeline crashed: {e}[/bold red]")
        sys.exit(1)
    finally:
        status.stop()

    # ── Handle result ─────────────────────────────────────────────────────────
    if result.success:
        logger.info(
            f"Pipeline SUCCESS — "
            f"elapsed={result.elapsed_seconds:.2f}s "
            f"steps={result.total_steps}"
        )
        console.print(
            f"\n[bold green]✅ Pipeline complete![/bold green]  "
            f"Elapsed: [bold]{result.elapsed_seconds:.2f}s[/bold]  "
            f"Steps: [bold]{result.total_steps}[/bold]\n"
        )

        console.print("[bold]Saving generated files...[/bold]")
        saved = save_output_files(result, project_dir, language, app_file_name)

        if not saved:
            logger.warning("No files extracted from Generate output. Retrying pipeline once.")
            retry_result = run_pipeline(dot_content, config=config, emitter=emitter)
            saved = save_output_files(retry_result, project_dir, language, app_file_name)
            if not saved:
                logger.error(
                    "No code blocks detected after retry. Raw preview: %r",
                    retry_result.context.get_string("Generate.output", "")[:800],
                )
                console.print("[bold red]❌ No code blocks were detected after one retry. Stopping.[/bold red]")

        if saved:
            console.print(
                f"\n[bold green]📦 Project '[cyan]{slug}[/cyan]' "
                f"saved to [cyan]{project_dir}[/cyan][/bold green]\n"
            )
            print_project_tree(project_dir)
        else:
            console.print(
                "[bold yellow]⚠ Pipeline succeeded but no files were saved. "
                "Check attractor_agent.log.[/bold yellow]"
            )

    else:
        logger.error(f"Pipeline FAILED: {result.error}")
        console.print(f"\n[bold red]❌ Pipeline failed:[/bold red] {result.error}")
        console.print(
            "[dim]→ Check [bold]attractor_agent.log[/bold] for full details[/dim]"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_cli()
