from attractor.pipeline.context import Context
from attractor.pipeline.events import EventEmitter
from attractor.pipeline.graph import Graph, Node
from attractor.pipeline.handlers.builtin import CodergenHandler


class DummyBackend:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, node: Node, context: Context) -> str:
        return self.response


def test_codergen_preserves_full_output_in_context():
    response = """```python
# filename: a.py
print('a')
```
```python
# filename: b.py
print('b')
```"""

    handler = CodergenHandler()
    context = Context()
    node = Node(id="Generate", label="Generate", prompt="Make code")
    graph = Graph(name="g")
    emitter = EventEmitter()

    outcome = handler.execute(
        node=node,
        context=context,
        graph=graph,
        emitter=emitter,
        codergen_backend=DummyBackend(response),
    )

    assert outcome.status.value == "success"
    assert context.get_string("Generate.output", "") == response
    assert context.get_string("Generate.raw_output", "") == response
