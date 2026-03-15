# 🚀 Attractor Agent

**Attractor Agent** is a cutting-edge, conversational AI developer that transforms high-level project ideas into fully functional, production-ready codebases. Powered by the **Attractor Pipeline Engine**, it handles the entire Software Development Life Cycle (SDLC) — from architecture planning and code generation to unit testing and automated self-healing loops.

---

## ✨ Key Features

- **💬 Conversational UI**: Simply describe what you want to build in plain English.
- **🏗️ Automated SDLC**: Automatically generates complex `.dot` pipeline graphs that orchestrate multiple LLM agents.
- **🔄 Self-Healing Loop**: Autonomous **Observe → Plan → Code → Test → Fix → Score → Converge** cycle to ensure 100% bug-free delivery.
- **📈 Live Progress Tracker**: Watch every stage in real-time with event streaming.
- **🖥️ Multi-Interface**: High-performance **CLI**, sleek **Web GUI**, and a robust **REST API**.
- **📋 Conformance Suite**: 100% spec-compliant engine following [NLSpec](attractor-spec.md) for routing, tiebreaks, and goal-gates.
- **🧪 Real Test Execution**: Physically executes generated code and tests in a sandbox.
- **📊 Satisfaction Scorer**: Quantitative LLM-as-a-judge scoring with configurable thresholds.
- **🗄️ Persistence Layer**: Production-ready storage supporting both **SQLite** and **MongoDB**.

---

## 🎮 Getting Started

### 📦 Installation

Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 🔑 Configuration

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_DEFAULT_MODEL=openrouter/free

# Database Selection (sqlite or mongodb)
ATTRACTOR_DB=sqlite
```

---

## 🚀 How to Use

### 📟 Method 1: Interactive CLI
For a terminal-based experience with loading spinners and rich formatting:

```bash
# Start interactive builder
python -m attractor_agent

# Run a specific pipeline directly
python -m attractor_agent examples/full_sdlc.dot
```

### 🌐 Method 2: Web Interface (GUI)
For a visual experience with a real-time code editor and human-in-the-loop review:

```bash
python -m attractor_agent --gui
```

### 🔌 Method 3: REST API (Headless)
For integrating Attractor into your own services:

```bash
python -m attractor_agent --api
```

#### API Endpoints (Section 9.5)
- `POST /api/v1/runs`: Start a new project from a prompt.
- `POST /pipelines`: Submit a raw `.dot` pipeline file.
- `GET /pipelines/{run_id}`: Monitor status and health (RUNNING, COMPLETED, FAILED).
- `GET /api/v1/runs/{run_id}/events`: SSE stream of real-time pipeline events.

---

## 🐳 Docker Deployment

The fastest way to stand up a production environment with MongoDB persistence:

1. **Configure Environment**: Set `OPENROUTER_API_KEY` in your shell.
2. **Launch Stack**:
   ```bash
   docker-compose up -d
   ```

---

## 🏗️ Project Architecture

### The Self-Healing SDLC Loop
Attractor pipelines often implement a gated feedback loop:
1. **Plan**: LLM outlines the implementation strategy.
2. **Generate**: Core code implementation.
3. **RunTests**: Executes code and captures diagnostics (return codes, tracebacks).
4. **Diagnose**: Analyzes test failures and pinpoints root causes.
5. **TargetedFix**: Generates focused patches based on diagnosis.
6. **Scorer**: Grades the fixed code (0-100).
7. **Converge**: Exits only when tests pass AND score ≥ 95.

### Persistence & Storage
- **Local**: SQLite (`attractor_runs.db`) is used by default.
- **Scale**: Enable MongoDB via `ATTRACTOR_DB=mongodb`.
- **Logs**: All run data is stored in `projects/{run_id}/`.

---

## 🧪 Testing

### Reliability Suite
Run the full conformance and engine test suite:
```bash
python -m pytest tests/
```

### Production Readiness
Run the comprehensive end-to-end production check (CLI, API, and Persistence):
```bash
python tests/test_production.py
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using the <b>Heaven</b>
</p>
