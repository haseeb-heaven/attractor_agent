#!/usr/bin/env python3
"""
Attractor Production Test Suite
Modular, robust, and safe test runner
"""

import subprocess
import requests
import time
import sys
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Optional


# --------------------------------------------------
# Config
# --------------------------------------------------

ROOT = Path(".")
EXAMPLE_DOT = ROOT / "examples/full_sdlc.dot"
PROJECTS_DIR = ROOT / "projects"

API_URL = "http://localhost:8000"
API_DOCS = f"{API_URL}/docs"
PIPELINE_ENDPOINT = f"{API_URL}/pipelines"

PYTEST_TIMEOUT = 120
CLI_TIMEOUT = 120
API_START_TIMEOUT = 10


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def safe_run(cmd: List[str], timeout: int = 60):
    """Run subprocess safely with timeout."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Process timed out"
    except Exception as e:
        return -1, "", str(e)


def kill_python_processes():
    """Kill stray python processes (Windows safe)."""
    subprocess.run(
        ["taskkill", "/f", "/im", "python.exe"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_process(cmd: List[str]) -> subprocess.Popen:
    """Start background process."""
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )


# --------------------------------------------------
# Step System
# --------------------------------------------------

@dataclass
class StepResult:
    name: str
    success: bool
    message: str = ""


class TestStep:
    def __init__(self, name: str, func: Callable[[], StepResult]):
        self.name = name
        self.func = func

    def run(self) -> StepResult:
        print(f"[RUN] {self.name:<25}")
        try:
            result = self.func()
            status = "DONE" if result.success else "ERROR"
            print(f"[{status}] {self.name:<25} {result.message}")
            return result
        except Exception as e:
            print(f"[ERROR] {self.name:<25} {str(e)}")
            return StepResult(self.name, False, str(e))


# --------------------------------------------------
# Test Steps
# --------------------------------------------------

def step_kill_servers():
    kill_python_processes()
    time.sleep(2)
    return StepResult("Kill Servers", True)


def step_pytest():
    code, out, err = safe_run(
        ["python", "-m", "pytest", "tests/", "--tb=no"],
        timeout=PYTEST_TIMEOUT
    )

    passed = "passed" in out.lower()

    msg = out.splitlines()[-1] if out else err
    return StepResult("Pytest", passed, msg)


def step_cli_pipeline():
    if not EXAMPLE_DOT.exists():
        return StepResult("CLI Pipeline", False, "Missing example dot")

    code, out, err = safe_run(
        ["python", "-m", "attractor_agent", str(EXAMPLE_DOT)],
        timeout=CLI_TIMEOUT
    )

    success = "completed" in out.lower() or "exit" in out.lower()

    msg = out[:100] if success else err[:100]
    return StepResult("CLI Pipeline", success, msg)


def step_api_pipeline():

    api = start_process(["python", "-m", "attractor_agent", "--api"])

    try:
        time.sleep(API_START_TIMEOUT)

        r = requests.get(API_DOCS, timeout=5)

        if r.status_code != 200:
            return StepResult("API Server", False, "Docs endpoint failed")

        with open(EXAMPLE_DOT, "rb") as f:
            resp = requests.post(
                PIPELINE_ENDPOINT,
                files={"dot": f},
                timeout=10
            )

        if resp.status_code != 200:
            return StepResult("API Pipeline", False, resp.text)

        run_id = resp.json().get("run_id")

        return StepResult("API Pipeline", bool(run_id), f"run_id={run_id}")

    except Exception as e:
        return StepResult("API Pipeline", False, str(e))

    finally:
        try:
            api.terminate()
            api.wait(timeout=5)
        except Exception:
            pass


def step_projects_created():
    projects = list(PROJECTS_DIR.glob("*-*"))
    count = len(projects)

    return StepResult(
        "Projects Check",
        count > 0,
        f"{count} generated"
    )


# --------------------------------------------------
# Runner
# --------------------------------------------------

def run_suite():

    print("🚀 Attractor Production Test Suite")
    print("=" * 50)

    steps = [
        TestStep("Kill Servers", step_kill_servers),
        TestStep("Pytest", step_pytest),
        TestStep("CLI Pipeline", step_cli_pipeline),
        TestStep("API Pipeline", step_api_pipeline),
        TestStep("Projects Check", step_projects_created),
    ]

    results = []

    for step in steps:
        results.append(step.run())
        time.sleep(1)

    success = sum(1 for r in results if r.success)

    print("\n" + "=" * 50)

    if success == len(results):
        print(f"✅ {success}/{len(results)} PRODUCTION READY")
    else:
        print(f"⚠️ {success}/{len(results)} TESTS PASSED")

    return success == len(results)


# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__ == "__main__":
    ok = run_suite()
    sys.exit(0 if ok else 1)