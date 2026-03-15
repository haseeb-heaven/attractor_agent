"""Local execution environment implementation (Section 4.2)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from attractor.agent.types import DirEntry, ExecResult, ExecutionEnvironment


class LocalExecutionEnvironment(ExecutionEnvironment):
    """Execution environment that runs tools on the local machine."""

    def __init__(self, working_dir: str | None = None):
        self._working_dir = Path(working_dir or os.getcwd()).absolute()
        self._is_windows = os.name == "nt"

    def initialize(self) -> None:
        """Initialize the environment (ensure working directory exists)."""
        self._working_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        """Cleanup (no-op for local)."""
        pass

    def working_directory(self) -> str:
        return str(self._working_dir)

    def platform(self) -> str:
        import sys
        return sys.platform

    def os_version(self) -> str:
        import platform
        return platform.version()

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self._working_dir / p
        return p.absolute()

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        full_path = self._resolve_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        start = (offset - 1) if offset else 0
        end = (start + limit) if limit else None
        
        selected_lines = lines[start:end]
        
        # Prepend line numbers (NNN | content)
        result = []
        for i, line in enumerate(selected_lines):
            line_num = start + i + 1
            result.append(f"{line_num:4} | {line}")
        
        return "".join(result)

    def read_file_raw(self, path: str) -> str:
        full_path = self._resolve_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def write_file(self, path: str, content: str) -> None:
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    def file_exists(self, path: str) -> bool:
        return self._resolve_path(path).exists()

    def list_directory(self, path: str, depth: int = 1) -> list[DirEntry]:
        root = self._resolve_path(path)
        if not root.exists() or not root.is_dir():
            return []
        
        entries = []
        for item in root.iterdir():
            entries.append(DirEntry(
                name=item.name,
                is_dir=item.is_dir(),
                size=item.stat().st_size if item.is_file() else None
            ))
        return entries

    def exec_command(
        self,
        command: str,
        timeout_ms: int,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        cwd = self._resolve_path(working_dir) if working_dir else self._working_dir
        
        # Environment variable filtering (Section 4.2)
        filtered_env = self._get_filtered_env(env_vars)
        
        start_time = time.time()
        
        # Platform-specific shell
        shell_cmd = ["cmd.exe", "/c", command] if self._is_windows else ["/bin/bash", "-c", command]
        
        try:
            # Use start_new_session for process group on Unix
            # and CREATE_NEW_PROCESS_GROUP on Windows
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if self._is_windows else 0
            
            process = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=filtered_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                start_new_session=not self._is_windows,
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout_ms / 1000)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                # Section 5.4: SIGTERM, wait 2s, SIGKILL
                if self._is_windows:
                    # Windows doesn't have process groups in the same way, but CREATE_NEW_PROCESS_GROUP helps
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
                else:
                    import signal
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(2)
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                
                stdout, stderr = process.communicate()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode if not timed_out else -1,
                timed_out=timed_out,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return ExecResult(
                stdout="",
                stderr=f"Exception during execution: {e}",
                exit_code=-1,
                timed_out=False,
                duration_ms=int((time.time() - start_time) * 1000)
            )

    def _get_filtered_env(self, extra_vars: dict[str, str] | None) -> dict[str, str]:
        env = os.environ.copy()
        if extra_vars:
            env.update(extra_vars)
            
        filtered = {}
        # Sensitive var patterns (Section 4.1)
        sensitive_patterns = [
            re.compile(p.replace("*", ".*"), re.IGNORECASE)
            for p in ["*_API_KEY", "*_SECRET", "*_TOKEN", "*_PASSWORD", "*_CREDENTIAL"]
        ]
        
        # Always include (Section 4.2)
        always_include = {
            "PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "TMPDIR",
            "GOPATH", "CARGO_HOME", "NVM_DIR", "PYTHONPATH"
        }
        
        for k, v in env.items():
            if k in always_include:
                filtered[k] = v
                continue
                
            is_sensitive = any(p.match(k) for p in sensitive_patterns)
            if not is_sensitive:
                filtered[k] = v
                
        return filtered

    def grep(self, pattern: str, path: str, case_insensitive: bool = False, max_results: int = 100) -> str:
        # Use ripgrep if available, otherwise fallback to simple python grep
        full_path = self._resolve_path(path)
        rg_path = shutil.which("rg")
        
        if rg_path:
            cmd = [rg_path, "--line-number", "--heading", "--max-count", str(max_results)]
            if case_insensitive:
                cmd.append("--ignore-case")
            cmd.append(pattern)
            cmd.append(str(full_path))
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                return result.stdout + result.stderr
            except Exception as e:
                return f"Error running ripgrep: {e}"
        
        # Fallback to simple python grep
        # ... (implementation omitted for brevity, but could be added if needed)
        return "ripgrep not found. Python-native grep fallback not yet fully implemented."

    def glob(self, pattern: str, path: str) -> list[str]:
        root = self._resolve_path(path)
        matches = list(root.glob(pattern))
        # Sort by mtime (Section 3.3 glob)
        matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        return [str(p.relative_to(self._working_dir)) for p in matches]
