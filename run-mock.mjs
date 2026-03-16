import http from "http";

const port = 5555;
const MAX_SIZE = 1_000_000;

const response = {
	id: "mock-123",
	object: "chat.completion",
	created: Date.now(),
	model: "mock-model",
	choices: [
		{
			index: 0,
			message: {
				role: "assistant",
				content: "```python\n# filename: main.py\nprint('hello from mock')\n```",
			},
			finish_reason: "stop",
		},
	],
	usage: {
		prompt_tokens: 10,
		completion_tokens: 20,
		total_tokens: 30,
	},
};

const server = http.createServer((req, res) => {
	if (req.method === "POST" && req.url === "/v1/chat/completions") {
		let body = "";
		req.on("data", (chunk) => {
			body += chunk;
			if (body.length > MAX_SIZE) {
				req.destroy();
			}
		});

		req.on("end", () => {
			res.writeHead(200, { "Content-Type": "application/json" });
			res.end(JSON.stringify(response));
		});
		return;
	}

	res.writeHead(404);
	res.end();
});

server.listen(port, () => {
	console.log("Mock OpenAI server running on port", port);
});
