from __future__ import annotations

from typing import Any

from attractor.agent.types import ExecutionEnvironment


def read_file_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    return env.read_file(
        path=args["file_path"],
        offset=args.get("offset"),
        limit=args.get("limit", 2000)
    )


def write_file_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    env.write_file(path=args["file_path"], content=args["content"])
    return f"Successfully wrote to {args['file_path']}"


def shell_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    result = env.exec_command(
        command=args["command"],
        timeout_ms=args.get("timeout_ms", 10000),
        description=args.get("description")
    )
    
    output = f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}\nExit Code: {result.exit_code}"
    if result.timed_out:
        output += f"\n\n[ERROR: Command timed out after {result.duration_ms}ms. Partial output is shown above. You can retry with a longer timeout by setting the timeout_ms parameter.]"
    
    return output


def grep_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    return env.grep(
        pattern=args["pattern"],
        path=args.get("path", "."),
        case_insensitive=args.get("case_insensitive", False),
        max_results=args.get("max_results", 100)
    )


def glob_executor(args: dict[str, Any], env: ExecutionEnvironment) -> str:
    matches = env.glob(
        pattern=args["pattern"],
        path=args.get("path", ".")
    )
    return "\n".join(matches)
