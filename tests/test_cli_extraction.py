from attractor_agent.cli import (
    _default_filename_for_language,
    extract_blocks_with_fallbacks,
)


def test_markdown_parser_takes_priority_over_fallback_markers():
    text = """```python file=main.py
# filename: main.py
print('ok')
```

START CODE file=ignored.py
print('fallback')
END CODE
"""
    blocks = extract_blocks_with_fallbacks(text)
    assert len(blocks) == 1
    assert blocks[0].attribute_filename == "main.py"
    assert "fallback" not in blocks[0].code


def test_header_section_fallback_splits_files():
    text = """=== index.html ===
<html><script>console.log('in html')</script></html>
=== styles.css ===
body { color: red; }
"""
    blocks = extract_blocks_with_fallbacks(text)
    assert len(blocks) == 2
    assert blocks[0].header_filename == "index.html"
    assert blocks[1].header_filename == "styles.css"


def test_filename_comment_fallback_splits_files():
    text = """// filename: app.js
console.log('a')
# filename: main.py
print('b')
"""
    blocks = extract_blocks_with_fallbacks(text)
    assert len(blocks) == 2
    assert blocks[0].filename_comment == "app.js"
    assert blocks[1].filename_comment == "main.py"


def test_default_filename_language_map():
    assert _default_filename_for_language("java", 0) == "Main.java"
    assert _default_filename_for_language("c#", 0) == "Program.cs"
    assert _default_filename_for_language("html", 0) == "index.html"
    assert _default_filename_for_language("css", 0) == "styles.css"
    assert _default_filename_for_language("javascript", 1) == "app_2.js"
