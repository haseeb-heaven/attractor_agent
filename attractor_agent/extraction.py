from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ExtractedBlock:
    """Structured extracted code block."""

    language: str
    code: str
    filename_comment: str | None = None
    header_filename: str | None = None
    attribute_filename: str | None = None


def _extract_filename_comment(text: str) -> str | None:
    patterns = [
        r"^\s*//\s*filename\s*:\s*(.+?)\s*$",
        r"^\s*#\s*filename\s*:\s*(.+?)\s*$",
        r"^\s*/\*\s*filename\s*:\s*(.+?)\s*\*/\s*$",
        r"^\s*<!--\s*filename\s*:\s*(.+?)\s*-->\s*$",
        r"^\s*filename\s*:\s*(.+?)\s*$",
    ]
    for line in text.splitlines()[:5]:
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return None


def _strip_filename_markers(text: str) -> str:
    lines = text.splitlines()
    start = 0
    while start < min(len(lines), 5) and _extract_filename_comment(lines[start]):
        start += 1
    return "\n".join(lines[start:]).strip()


def _parse_info_attributes(info: str) -> tuple[str, str | None]:
    if not info.strip():
        return "", None

    parts = info.strip().split()
    language = parts[0]
    attribute_filename = None
    patterns = [
        r"(?:file|filename|name)\s*=\s*\"([^\"]+)\"",
        r"(?:file|filename|name)\s*=\s*'([^']+)'",
        r"(?:file|filename|name)\s*=\s*([^\s]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, info, re.IGNORECASE)
        if match:
            attribute_filename = match.group(1).strip()
            break
    return language, attribute_filename


def _extract_markdown_blocks(text: str) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    pattern = re.compile(r"```([^\n]*)\r?\n(.*?)\r?\n```", re.DOTALL)
    for info, code in pattern.findall(text):
        cleaned = code.strip()
        if not cleaned:
            continue
        language, attribute_filename = _parse_info_attributes(info)
        filename_comment = _extract_filename_comment(cleaned)
        blocks.append(
            ExtractedBlock(
                language=language.lower().strip(),
                code=_strip_filename_markers(cleaned),
                filename_comment=filename_comment,
                attribute_filename=attribute_filename,
            )
        )
    return blocks


def _extract_fallback_marker_blocks(text: str) -> list[ExtractedBlock]:
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
    current_language = ""
    current_attribute_filename: str | None = None
    inside_block = False

    def flush() -> None:
        if not current_lines:
            return
        code = "\n".join(current_lines).strip()
        if not code:
            return
        blocks.append(
            ExtractedBlock(
                language=current_language,
                code=_strip_filename_markers(code),
                filename_comment=_extract_filename_comment(code),
                attribute_filename=current_attribute_filename,
            )
        )

    for line in text.splitlines():
        start_match = start_re.match(line)
        if start_match:
            if inside_block:
                flush()
            current_lines = []
            inside_block = True
            current_language, current_attribute_filename = _parse_info_attributes(
                start_match.group(1).strip()
            )
            continue
        if inside_block and end_re.match(line):
            flush()
            current_lines = []
            current_language = ""
            current_attribute_filename = None
            inside_block = False
            continue
        if inside_block:
            current_lines.append(line)

    if inside_block:
        flush()
    return [block for block in blocks if block.code]


def _extract_header_sections(text: str) -> list[ExtractedBlock]:
    header_re = re.compile(
        r"^\s*(?:===\s*(.+?)\s*===|---\s*(.+?)\s*---|###\s+(.+?)\s*|FILE\s*:\s*(.+?)\s*)$",
        re.IGNORECASE,
    )
    blocks: list[ExtractedBlock] = []
    current_filename: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = header_re.match(line)
        if match:
            if current_filename and current_lines:
                code = "\n".join(current_lines).strip()
                if code:
                    blocks.append(
                        ExtractedBlock(
                            language="",
                            code=code,
                            header_filename=current_filename,
                        )
                    )
            current_filename = next((group.strip() for group in match.groups() if group), None)
            current_lines = []
            continue
        if current_filename:
            current_lines.append(line)

    if current_filename and current_lines:
        code = "\n".join(current_lines).strip()
        if code:
            blocks.append(
                ExtractedBlock(language="", code=code, header_filename=current_filename)
            )
    return blocks


def _extract_filename_comment_sections(text: str) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    current_filename: str | None = None
    current_lines: list[str] = []
    html_mode = False
    script_depth = 0

    for line in text.splitlines():
        lower_line = line.lower()
        if "<html" in lower_line or "<!doctype html" in lower_line:
            html_mode = True
        if html_mode and "<script" in lower_line:
            script_depth += 1

        marker = None if (html_mode and script_depth > 0) else _extract_filename_comment(line)
        if marker:
            if current_filename and current_lines:
                blocks.append(
                    ExtractedBlock(
                        language="",
                        code="\n".join(current_lines).strip(),
                        filename_comment=current_filename,
                    )
                )
            current_filename = marker
            current_lines = []
            continue

        if current_filename:
            current_lines.append(line)

        if html_mode and "</script>" in lower_line and script_depth > 0:
            script_depth -= 1

    if current_filename and current_lines:
        blocks.append(
            ExtractedBlock(
                language="",
                code="\n".join(current_lines).strip(),
                filename_comment=current_filename,
            )
        )
    return [block for block in blocks if block.code]


def extract_blocks_with_fallbacks(text: str) -> list[ExtractedBlock]:
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


def infer_language_from_filename(filename: str) -> str:
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
