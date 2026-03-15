# 🚀 Attractor Agent

**Attractor Agent** is a cutting-edge, conversational AI developer that transforms high-level project ideas into fully functional, production-ready codebases. Powered by the **Attractor Pipeline Engine**, it handles the entire Software Development Life Cycle (SDLC) — from architecture planning and code generation to unit testing and automated deployment.

---

## ✨ Key Features

- **💬 Conversational UI**: Simply describe what you want to build in plain English.
- **🏗️ Automated SDLC**: Automatically generates complex `.dot` pipeline graphs that orchestrate multiple LLM agents.
- **📈 Live Progress Tracker**: Watch every step (Plan → Generate → Test → Score → Deploy) in real-time.
- **🖥️ Dual Interface**: Choose between a high-performance **CLI** or a sleek **Web GUI**.
- **📋 Conformance Suite**: Rigorous validation ensuring the engine follows [NLSpec](attractor-spec.md) for routing and retries.
- **🧪 Real Test Execution**: Physically runs generated unit tests (`pytest`, `npm test`, etc.) in a sandbox to verify code.
- **📊 Satisfaction Scorer**: Quantitative LLM-as-a-judge scoring of generated artifacts against goals.
- **🗄️ Persistence Layer**: Production-ready storage supporting both **SQLite** and **MongoDB**.
- **🎭 LLMock Automation**: Full zero-config support for local, deterministic testing with [LLMock](https://llmock.copilotkit.dev/).

---

## 🎮 Getting Started

### 📦 Installation

Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 🔑 Configuration

Create a `.env` file in the root directory and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_DEFAULT_MODEL=openrouter/free

# Database Selection (sqlite or mongodb)
ATTRACTOR_DB=sqlite
```

---

## 🚀 How to Use

### 📟 Method 1: Interactive CLI
For a fast, terminal-based experience with loading spinners and rich formatting:

```bash
python -m attractor_agent
```

### 🌐 Method 2: Web Interface (GUI)
For a visual experience with a centered layout and real-time code editor:

```bash
python -m attractor_agent --gui --port 7860
```

### 🔌 Method 3: REST API (Headless)
For integrating Attractor into your own services or running as a backend:

```bash
python -m attractor_agent --api --port 8000
```
Swagger documentation will be available at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

The fastest way to stand up a production environment with MongoDB persistence:

1. **Configure Environment**: Set `OPENROUTER_API_KEY` in your shell.
2. **Launch Stack**:
   ```bash
   docker-compose up -d
   ```
This starts the **Attractor API** on port `8000` and a **MongoDB** instance for persistence.

---

## 🏗️ Project Architecture

### The Advanced SDLC Pipeline
Attractor uses a state-of-the-art graph-based execution model:
1. **Plan Architecture**: LLM outlines the structure and file signatures.
2. **Generate Code**: Core implementation logic.
3. **Unit Tests**: Generates comprehensive testing suites.
4. **Run Tests**: **(Physical Execution)** Runs the scripts in a sandbox. Fails if code is buggy.
5. **Satisfaction Score**: Quantitative grading of the output (0-100).
6. **SDLC Validation**: Technical review for security and robust error handling.
7. **Human Review**: Final inspection and approval.
8. **Digital Twin**: Registers finalized code into the mock deployment "Universe".

### Persistence & Storage
Runs and events are automatically persisted to the database layer.
- **SQLite**: Default for local development (`attractor_runs.db`).
- **MongoDB**: Used for production scale (enabled via `ATTRACTOR_DB=mongodb`).

---

## 🧪 Running Tests

To verify the Attractor engine follows the NLSpec strictly, run the conformance suite:

```bash
python -m pytest tests/test_conformance.py
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using the <b>Heaven</b>
</p>
