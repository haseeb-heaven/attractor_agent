# Attractor Agent

Attractor Agent turns a build request into an autonomous SDLC pipeline: plan, generate, test, score, optionally review, and emit deployment artifacts. The CLI, GUI, and API now share the same request model and can all load build parameters from external JSON or TOML files.

## Key capabilities

- Autonomous mode by default. Human review is optional instead of mandatory.
- Headless CLI execution with `--request` flags or `--config` files.
- Shared GUI and CLI config flow. Both consume the same external build file format.
- Consistent mock mode. CLI and GUI both use the local `run-mock.mjs` server.
- Deployment artifact generation in the `digital_twin` stage.
- GitHub Actions test workflow for pull requests.

## Install

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

Set your LLM credentials in `.env` when you are not using the mock server.

## CLI

Interactive mode:

```bash
python -m attractor_agent
```

Headless mode:

```bash
python -m attractor_agent --request "Build a Flask books app" --language Python --framework Flask
python -m attractor_agent --config examples/autonomous_build.json
```

To force a human review gate:

```bash
python -m attractor_agent --config examples/autonomous_build.json --require-human-review --no-auto-approve
```

## GUI

```bash
python -m attractor_agent --gui
```

The GUI accepts direct form input or a `.json` / `.toml` config upload and runs the same build flow as the CLI.

## API

```bash
python -m attractor_agent --api
```

Core endpoints:

- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/runs/{run_id}/events`
- `GET /api/v1/runs/{run_id}/questions`
- `POST /api/v1/runs/{run_id}/questions/{qid}/answer`
- `POST /pipelines`

## Config file

Example: [examples/autonomous_build.json](examples/autonomous_build.json)

Supported keys:

- `request`
- `language`
- `framework`
- `include_tests`
- `include_sdlc`
- `use_mock`
- `auto_approve`
- `require_human_review`
- `retry_save_attempts`
- `project_name`

## Testing

```bash
pytest tests/
```

## License

MIT. See [LICENSE](LICENSE).
