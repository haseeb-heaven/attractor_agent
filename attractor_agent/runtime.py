from __future__ import annotations

import atexit
import json
import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from attractor.pipeline.backend import LLMBackend
from attractor.pipeline.engine import PipelineConfig, PipelineResult, run_pipeline
from attractor.pipeline.events import EventEmitter
from attractor.pipeline.interviewer import AutoApproveInterviewer, Interviewer

from attractor_agent.extraction import (
    ExtractedBlock,
    extract_blocks_with_fallbacks,
    infer_language_from_filename,
)
from attractor_agent.project import BuildRequest, build_dot, get_extension

logger = logging.getLogger("attractor_agent.runtime")


def wait_for_port(host: str, port: int, timeout: float = 45.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Server at {host}:{port} did not respond within {timeout} seconds.")


class MockServer:
    def __init__(self, command: list[str] | None = None):
        script = Path(__file__).resolve().parent / "run-mock.mjs"
        self.command = command or ["node", str(script)]
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.process = subprocess.Popen(self.command, stdout=None, stderr=None)
        atexit.register(self.stop)
        wait_for_port("127.0.0.1", 5555, timeout=45.0)

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()


def create_backend(use_mock: bool) -> tuple[LLMBackend, Callable[[], None]]:
    if not use_mock:
        return LLMBackend(), lambda: None

    server = MockServer()
    server.start()

    from attractor.llm.adapters.litellm import LiteLLMAdapter
    from attractor.llm.client import Client

    mock_adapter = LiteLLMAdapter(
        api_key="mock-key",
        api_base="http://127.0.0.1:5555/v1",
        default_model="openai/gpt-4o-mini",
    )
    client = Client(providers={"mock": mock_adapter}, default_provider="mock")
    return LLMBackend(client=client), server.stop


def get_project_dir(spec: BuildRequest) -> Path:
    if spec.checkpoint_dir:
        return Path(spec.checkpoint_dir)
    return Path("projects") / spec.normalized_project_slug()


def get_file_extension_from_tag(lang_tag: str, fallback_language: str) -> str:
    ext_map = {
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "html": ".html",
        "css": ".css",
        "sql": ".sql",
        "python": ".py",
        "go": ".go",
        "rust": ".rs",
        "java": ".java",
        "cpp": ".cpp",
        "c++": ".cpp",
        "csharp": ".cs",
        "c#": ".cs",
        "bash": ".sh",
        "shell": ".sh",
        "json": ".json",
        "yaml": ".yml",
        "yml": ".yml",
        "markdown": ".md",
        "md": ".md",
        "xml": ".xml",
        "toml": ".toml",
    }
    return ext_map.get(lang_tag.lower().strip(), get_extension(fallback_language))


def get_smart_filename(lang_tag: str, index: int, language: str) -> str:
    ext = get_file_extension_from_tag(lang_tag, language)
    name_map: dict[str, list[str]] = {
        ".html": ["index.html", "login.html", "register.html", "dashboard.html"],
        ".css": ["style.css", "app.css", "components.css"],
        ".js": ["main.js", "app.js", "api.js", "utils.js"],
        ".ts": ["main.ts", "app.ts", "api.ts", "utils.ts"],
        ".sql": ["schema.sql", "seed.sql", "migrations.sql"],
        ".py": ["main.py", "app.py", "models.py", "utils.py", "routes.py"],
        ".sh": ["setup.sh", "run.sh", "build.sh"],
        ".json": ["package.json", "config.json", "tsconfig.json"],
        ".java": ["Main.java", "App.java", "Utils.java"],
        ".cpp": ["main.cpp", "app.cpp", "utils.cpp"],
        ".go": ["main.go", "app.go", "utils.go"],
    }
    names = name_map.get(ext, [])
    if index < len(names):
        return names[index]
    return f"file_{index + 1}{ext}"


def _default_filename_for_language(lang: str, index: int) -> str:
    normalized = lang.lower().strip()
    if normalized in {"java"}:
        return "Main.java"
    if normalized in {"csharp", "c#"}:
        return "Program.cs"
    if normalized in {"go"}:
        return "main.go"
    if normalized in {"cpp", "c++"}:
        return "main.cpp"
    if normalized in {"rust"}:
        return "main.rs"
    if normalized in {"javascript", "js"}:
        return "main.js" if index == 0 else f"app_{index + 1}.js"
    if normalized in {"typescript", "ts"}:
        return "main.ts" if index == 0 else f"app_{index + 1}.ts"
    if normalized in {"html"}:
        return "index.html"
    if normalized in {"css"}:
        return "styles.css"
    if normalized in {"sql"}:
        return "schema.sql"
    if normalized in {"python", "py"}:
        return "main.py" if index == 0 else f"app_{index + 1}.py"
    if normalized in {"bash", "shell", "sh"}:
        return "run.sh"
    if normalized in {"json"}:
        return "config.json"
    return f"file_{index + 1}.txt"


def _resolve_filename(block: ExtractedBlock, index: int, language: str) -> str:
    for candidate in (
        block.attribute_filename,
        block.filename_comment,
        block.header_filename,
    ):
        if candidate:
            return candidate.strip()

    if block.language:
        return get_smart_filename(block.language, index, language)
    return _default_filename_for_language(language, index)


def _safe_relative_path(filename: str) -> Path | None:
    raw = Path(filename)
    if raw.is_absolute():
        return None
    # Block Windows drive prefixes like C:\...
    if raw.drive or (raw.parts and raw.parts[0].endswith(":")):
        return None
    normalized = Path(*[part for part in raw.parts if part not in ("", ".")])
    if any(part == ".." for part in normalized.parts):
        return None
    return normalized


def _readme_run_instructions(language: str) -> str:
    run_instructions = {
        "python": "```bash\npip install -r requirements.txt\npython main.py\n```",
        "javascript": "```bash\nnpm install\nnode main.js\n```",
        "typescript": "```bash\nnpm install\nnpx ts-node main.ts\n```",
        "java": "```bash\njavac Main.java\njava Main\n```",
        "c++": "```bash\ng++ -std=c++17 -o app main.cpp\n./app\n```",
        "go": "```bash\ngo run main.go\n```",
        "rust": "```bash\ncargo run\n```",
        "html/css": "Open `index.html` in a browser.",
        "html": "Open `index.html` in a browser.",
    }
    return run_instructions.get(language.lower(), "See the language documentation for run instructions.")


def save_output_files(
    result: PipelineResult,
    project_dir: Path,
    language: str,
    app_file_name: str,
) -> list[Path]:
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

    if not generate_output:
        return []

    extracted_blocks = extract_blocks_with_fallbacks(generate_output)
    if not extracted_blocks:
        return []

    project_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    extracted_manifest: list[dict[str, str]] = []
    for index, block in enumerate(extracted_blocks):
        code = block.code.strip()
        if not code:
            continue
        filename = _resolve_filename(block, index, language)
        safe_rel = _safe_relative_path(filename)
        if safe_rel is None:
            logger.warning("Skipped unsafe filename from model output: %s", filename)
            continue
        safe_path = (project_dir / safe_rel).resolve()
        project_root = project_dir.resolve()
        if project_root != safe_path and project_root not in safe_path.parents:
            logger.warning("Skipped unsafe resolved filename from model output: %s", filename)
            continue
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(code, encoding="utf-8")
        saved_files.append(safe_path)
        extracted_manifest.append(
            {
                "filename": safe_rel.as_posix(),
                "language": block.language or infer_language_from_filename(safe_rel.name) or language.lower(),
                "content": code,
            }
        )

    if not saved_files:
        return []

    result.context.set("Generate.extracted_files", extracted_manifest)

    if tests_output:
        test_blocks = extract_blocks_with_fallbacks(tests_output)
        test_code = test_blocks[0].code.strip() if test_blocks else tests_output.strip()
        test_file = project_dir / f"test_main{get_extension(language)}"
        test_file.write_text(test_code, encoding="utf-8")
        saved_files.append(test_file)

    readme_lines = [
        "# Generated Project\n\n",
        "**Built by:** Attractor Agent\n",
        f"**Language:** {language}\n\n",
        "## Project Files\n\n",
    ]
    for file_path in saved_files:
        readme_lines.append(f"- `{file_path.relative_to(project_dir).as_posix()}`\n")
    readme_lines.append(f"\n## How to Run\n\n{_readme_run_instructions(language)}\n")
    (project_dir / "README.md").write_text("".join(readme_lines), encoding="utf-8")

    return saved_files


@dataclass(slots=True)
class ExecutionArtifacts:
    spec: BuildRequest
    project_dir: Path
    dot_file: Path
    result: PipelineResult
    saved_files: list[Path]
    attempts_used: int


def write_build_metadata(spec: BuildRequest, project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "build_request.json").write_text(spec.to_json(), encoding="utf-8")


def execute_build(
    spec: BuildRequest,
    interviewer: Interviewer | None = None,
    emitter: EventEmitter | None = None,
) -> ExecutionArtifacts:
    emitter = emitter or EventEmitter()
    project_dir = get_project_dir(spec)
    project_dir.mkdir(parents=True, exist_ok=True)
    write_build_metadata(spec, project_dir)

    backend, cleanup = create_backend(spec.use_mock)
    try:
        active_interviewer = interviewer
        if active_interviewer is None and spec.auto_approve and not spec.require_human_review:
            active_interviewer = AutoApproveInterviewer()

        final_result: PipelineResult | None = None
        saved_files: list[Path] = []
        attempts_used = 0

        for attempt in range(spec.retry_save_attempts):
            attempts_used = attempt + 1
            dot_content = build_dot(spec, attempt=attempt)
            dot_file = project_dir / "pipeline.dot"
            dot_file.write_text(dot_content, encoding="utf-8")

            config = PipelineConfig(
                simulate=False,
                checkpoint_dir=str(project_dir),
                goal=spec.request,
                interviewer=active_interviewer,
                codergen_backend=backend,
            )
            final_result = run_pipeline(dot_content, config=config, emitter=emitter)

            if not final_result.success:
                break

            saved_files = save_output_files(
                final_result,
                project_dir,
                spec.language,
                spec.app_file_name(),
            )
            if saved_files:
                break

            logger.warning("No structured files extracted on attempt %s/%s", attempts_used, spec.retry_save_attempts)

        if final_result is None:
            raise RuntimeError("Pipeline did not execute.")
        if final_result.success and not saved_files:
            raise RuntimeError("No files were extracted from model output.")

        return ExecutionArtifacts(
            spec=spec,
            project_dir=project_dir,
            dot_file=project_dir / "pipeline.dot",
            result=final_result,
            saved_files=saved_files,
            attempts_used=attempts_used,
        )
    finally:
        cleanup()
