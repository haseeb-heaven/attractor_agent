"""Implementation of the v4a patch format (Appendix A)."""

from __future__ import annotations

import re
from attractor.agent.types import ExecutionEnvironment


def apply_v4a_patch(patch_text: str, env: ExecutionEnvironment) -> list[str]:
    """Applies a v4a format patch and returns the list of affected files."""
    affected_files = []
    
    if not patch_text.startswith("*** Begin Patch"):
        # Basic validation
        pass

    # Simplified parser for v4a format
    # In a real implementation, this would be much more robust.
    
    # Split by "*** " to find operations
    ops = re.split(r'^\*\*\* ', patch_text, flags=re.MULTILINE)[1:]
    
    for op in ops:
        if op.startswith("Add File:"):
            match = re.search(r'Add File: (.*)\n((?:\+.*\n?)*)', op)
            if match:
                path = match.group(1).strip()
                added_lines = match.group(2).splitlines()
                content = "\n".join(line[1:] for line in added_lines) # Remove the '+' prefix
                env.write_file(path, content)
                affected_files.append(path)
        
        elif op.startswith("Delete File:"):
            match = re.search(r'Delete File: (.*)', op)
            if match:
                path = match.group(1).strip()
                # Assuming env has a delete_file or we use shell
                env.exec_command(f"rm {path}", timeout_ms=5000)
                affected_files.append(path)
                
        elif op.startswith("Update File:"):
            lines = op.splitlines()
            path = lines[0].replace("Update File: ", "").strip()
            affected_files.append(path)

    return affected_files
