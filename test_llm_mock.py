import subprocess
import os
from pathlib import Path

# Create launcher script (bypasses CLI bug)
script = Path("run-mock.mjs")
script.write_text(f'''
import {{ LLMock }} from "@copilotkit/llmock";
import path from "path";

const mock = new LLMock({{ port: 5555 }});
mock.loadFixtureDir(path.resolve(".", "fixtures"));
await mock.start();
''')

# Run it
process = subprocess.Popen(
    ["node", "run-mock.mjs"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("Mock running on http://localhost:5555")
