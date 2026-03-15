from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_LANGUAGES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "HTML/CSS",
    "Go",
    "Rust",
    "C++",
    "Java",
]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", text.lower())
    return re.sub(r"_+", "_", slug).strip("_")


def get_extension(language: str) -> str:
    lang = language.lower()
    mapping = {
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "html": ".html",
        "html/css": ".html",
        "go": ".go",
        "rust": ".rs",
        "c++": ".cpp",
        "cpp": ".cpp",
        "java": ".java",
    }
    for key, value in mapping.items():
        if key in lang:
            return value
    return ".py"


def get_gradio_language(language: str) -> str:
    lang = language.lower()
    if "javascript" in lang:
        return "javascript"
    if "typescript" in lang:
        return "typescript"
    if "html" in lang:
        return "html"
    if "go" in lang:
        return "go"
    if "rust" in lang:
        return "rust"
    if "c++" in lang or "cpp" in lang:
        return "cpp"
    if "java" in lang and "script" not in lang:
        return "java"
    return "python"


@dataclass(slots=True)
class BuildRequest:
    request: str
    language: str = "Python"
    framework: str = ""
    include_tests: bool = True
    include_sdlc: bool = True
    use_mock: bool = False
    auto_approve: bool = True
    require_human_review: bool = False
    retry_save_attempts: int = 3
    project_name: str = ""
    checkpoint_dir: str = ""

    def normalized_project_slug(self) -> str:
        base = self.project_name or self.request
        return slugify(base)[:40] or "project"

    def app_file_name(self) -> str:
        return f"main{get_extension(self.language)}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_request_from_mapping(data: dict[str, Any]) -> BuildRequest:
    request = str(data.get("request") or data.get("prompt") or "").strip()
    if not request:
        raise ValueError("Build config must include 'request' or 'prompt'.")

    return BuildRequest(
        request=request,
        language=str(data.get("language") or "Python").strip() or "Python",
        framework=str(data.get("framework") or "").strip(),
        include_tests=_as_bool(data.get("include_tests"), True),
        include_sdlc=_as_bool(data.get("include_sdlc"), True),
        use_mock=_as_bool(data.get("use_mock") or data.get("mock_llm"), False),
        auto_approve=_as_bool(data.get("auto_approve"), True),
        require_human_review=_as_bool(
            data.get("require_human_review") or data.get("human_review"),
            False,
        ),
        retry_save_attempts=max(1, int(data.get("retry_save_attempts") or 3)),
        project_name=str(data.get("project_name") or "").strip(),
        checkpoint_dir=str(data.get("checkpoint_dir") or "").strip(),
    )


def load_build_request(path: str | Path) -> BuildRequest:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(raw)
    elif suffix == ".toml":
        data = tomllib.loads(raw)
    else:
        raise ValueError("Unsupported config format. Use .json or .toml.")

    if not isinstance(data, dict):
        raise ValueError("Build config root must be an object/table.")
    return build_request_from_mapping(data)


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _retry_directive(attempt: int) -> str:
    directives = [
        "",
        "\\nRetry guidance:\\n- Previous output could not be extracted reliably\\n- Return only code blocks\\n- Every file must start with an explicit filename marker",
        "\\nRetry guidance:\\n- Do not include prose, summaries, headings, or bullet points\\n- Use one fenced code block per file\\n- Use exact filenames and valid language tags",
        "\\nRetry guidance:\\n- Treat formatting compliance as mandatory\\n- If unsure, output fewer files but ensure every block is well-formed and complete",
    ]
    index = min(attempt, len(directives) - 1)
    return directives[index]


def build_dot(spec: BuildRequest, attempt: int = 0) -> str:
    framework_text = f" using {spec.framework}" if spec.framework else ""
    goal = _escape_dot(f"Build: {spec.request} in {spec.language}{framework_text}")
    request = _escape_dot(spec.request)
    language = _escape_dot(spec.language)
    framework = _escape_dot(spec.framework)
    framework_clause = f"using {framework}" if framework else ""
    retry_directive = _retry_directive(attempt)

    sections = [
        "digraph generate_app {",
        "    rankdir=LR;",
        f'    goal="{goal}";',
        "",
        '    node [shape=box, timeout="1200s", max_retries=2]',
        "",
        '    Start [shape=Mdiamond, label="Start"];',
        '    Done  [shape=Msquare,  label="Done"];',
        "",
        "    Plan [",
        '        label="Plan Architecture",',
        f'        prompt="Plan architecture for: {request} in {language} {framework_clause}.\\nOutput:\\n- Full folder and file structure with exact filenames\\n- Each module or file purpose\\n- All dependencies\\n- Database schema if needed\\nNO CODE YET.",',
        "        goal_gate=true",
        "    ];",
        "",
        "    Generate [",
        '        label="Generate Code",',
        f'        prompt="Using the plan above, write COMPLETE {language} code for: {request} {framework_clause}.\\nStrict rules:\\n- Output EVERY file as a SEPARATE triple backtick code block\\n- Label EVERY block with a valid language tag\\n- First line inside EVERY block must be a filename comment such as // filename: path/to/file.js or # filename: path/to/file.py\\n- ALL files must be complete, functional, and runnable\\n- No explanations or text between blocks{retry_directive}",',
        "        goal_gate=true",
        "    ];",
        "",
    ]

    if spec.include_tests:
        sections.extend(
            [
                "    Tests [",
                '        label="Unit Tests",',
                f'        prompt="Write COMPLETE unit tests for ALL functions in the generated code using {language}.\\n- Cover edge cases\\n- Use the standard test framework for {language}\\n- Output ONLY test code in a single labeled triple backtick block\\n- First line: // filename: test_main{get_extension(spec.language)}",',
                "        goal_gate=true",
                "    ];",
                "",
                '    RunTests [handler="test_runner", label="Execute Tests", max_retries=2];',
                "",
            ]
        )

    if spec.include_sdlc:
        sections.extend(
            [
                "    SDLCCheck [",
                '        label="SDLC Validation",',
                '        prompt="Review ALL generated code files for:\\n1. Error handling and exceptions\\n2. Input validation and sanitization\\n3. Security issues including SQL injection, XSS, and auth checks\\n4. Logging and observability\\n5. Code documentation\\n6. Memory and performance issues\\nOutput PASS if everything is good, or a numbered list of specific issues per file.",',
                "        goal_gate=true",
                "    ];",
                "",
            ]
        )

    sections.extend(
        [
            '    Score [handler="satisfaction_scorer", label="Satisfaction Scorer"];',
            '    DeployTwin [handler="digital_twin", label="Deploy Artifacts"];',
        ]
    )

    if spec.require_human_review:
        sections.extend(
            [
                "    Review [",
                '        shape=hexagon,',
                '        label="Human Review",',
                '        prompt="Review the generated output below and choose whether to approve deployment or send it back for another generation pass.\\n\\n${Generate.output}"',
                "    ];",
            ]
        )

    sections.extend(["", "    Start -> Plan -> Generate;"])

    if spec.include_tests:
        sections.extend(
            [
                "    Generate -> Tests -> RunTests;",
                '    RunTests -> Score [label="pass", condition="outcome=success"];',
                '    RunTests -> Generate [label="retry_code", condition="outcome=fail"];',
            ]
        )
    else:
        sections.append("    Generate -> Score;")

    if spec.include_sdlc:
        sections.extend(
            [
                "    Score -> SDLCCheck;",
                '    SDLCCheck -> Generate [condition="outcome=fail", label="[F]ix"];',
            ]
        )
        sections.append(
            "    SDLCCheck -> Review;" if spec.require_human_review else "    SDLCCheck -> DeployTwin;"
        )
    else:
        sections.append("    Score -> Review;" if spec.require_human_review else "    Score -> DeployTwin;")

    if spec.require_human_review:
        sections.extend(
            [
                '    Review -> DeployTwin [label="[A]pprove"];',
                '    Review -> Generate [label="[R]etry"];',
            ]
        )

    sections.extend(["    DeployTwin -> Done;", "}"])
    return "\n".join(sections) + "\n"
