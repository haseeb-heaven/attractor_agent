"""Project document discovery and loading (Section 6.5)."""

from __future__ import annotations

from pathlib import Path
from attractor.agent.types import ExecutionEnvironment


def discover_project_docs(env: ExecutionEnvironment, provider_id: str) -> str:
    """Discover and load relevant project documentation (Section 6.5)."""
    working_dir = Path(env.working_directory())
    docs = []
    
    # Files to look for based on provider
    provider_docs = {
        "openai": [".codex/instructions.md"],
        "anthropic": ["CLAUDE.md"],
        "gemini": ["GEMINI.md"],
    }
    
    targets = ["AGENTS.md"] + provider_docs.get(provider_id, [])
    
    # Walk up to find git root or just look in working dir
    # For now, just look in working dir and its parent
    search_dirs = [working_dir, working_dir.parent]
    
    seen = set()
    total_bytes = 0
    max_bytes = 32768 # 32KB
    
    for d in search_dirs:
        for filename in targets:
            p = d / filename
            if p.exists() and p.is_file() and p not in seen:
                content = p.read_text(encoding="utf-8", errors="replace")
                
                if total_bytes + len(content) > max_bytes:
                    docs.append(f"--- {filename} (truncated) ---\n" + content[:max_bytes - total_bytes])
                    docs.append("[Project instructions truncated at 32KB]")
                    return "\n\n".join(docs)
                
                docs.append(f"--- {filename} ---\n{content}")
                total_bytes += len(content)
                seen.add(p)
                
    return "\n\n".join(docs)
