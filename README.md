# 🚀 Attractor Agent

**Attractor Agent** is a cutting-edge, conversational AI developer that transforms high-level project ideas into fully functional, production-ready codebases. Powered by the **Attractor Pipeline Engine**, it handles the entire Software Development Life Cycle (SDLC) — from architecture planning and code generation to unit testing and security validation.

---

## ✨ Key Features

- **💬 Conversational UI**: Simply describe what you want to build in plain English.
- **🏗️ Automated SDLC**: Automatically generates complex `.dot` pipeline graphs that orchestrate multiple LLM agents.
- **📈 Live Progress Tracker**: Watch every step of the process (Plan → Generate → Test → Review) in real-time.
- **🖥️ Dual Interface**: Choose between a high-performance **CLI** or a sleek, centered **Web GUI**.
- **🔍 Human-in-the-Loop**: Preview generated code and approve, fix, or retry at any stage.
- **🧪 Robust Testing**: Integrated support for automated unit test generation and verification.
- **🛡️ SDLC Validation**: Built-in review gates for error handling, security, and documentation.
- **🎭 LLMock Integration**: Support for [LLMock](https://llmock.copilotkit.dev/) to run deterministic, cost-free tests locally.

---

## 🎮 Getting Started

### 📦 Installation

Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
pip install rich gradio
```

### 🔑 Configuration

Create a `.env` file in the root directory and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_DEFAULT_MODEL=openrouter/free
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
python -m attractor_agent --gui
```
*The GUI will launch at `http://localhost:8000`.*

---

## 🏗️ Project Architecture

When you build a project, Attractor Agent creates a structural workspace in the `projects/` directory:

```text
projects/
└── your-cool-project/
    ├── pipeline.dot      # The generated SDLC graph
    ├── main.py           # The finalized source code
    └── (logs/checkpoints)
```

### The SDLC Pipeline
Attractor uses a state-of-the-art graph-based execution model:
1. **Plan Architecture**: LLM outlines the structure and signatures.
2. **Generate Code**: Core logic implementation based on the plan.
3. **Unit Tests**: Generates comprehensive testing suites.
4. **Test Gate**: Validates that tests pass before moving forward.
5. **Human Review**: You inspect the code in the GUI/CLI.
6. **SDLC Validation**: Final polish for security and robust error handling.
7. **Mock Testing**: (Optional) Use LLMock to simulate model responses without API costs.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using the <b>Heaven</b>
</p>
