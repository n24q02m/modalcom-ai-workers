import re

file_path = "src/ai_workers/workers/asr.py"

with open(file_path, "r") as f:
    content = f.read()

# Fix result.get()
content = content.replace('text = result.get("text", "").strip()', 'text = result.get("text", "").strip()  # type: ignore')
content = content.replace('chunks = result.get("chunks", [])', 'chunks = result.get("chunks", [])  # type: ignore')

with open(file_path, "w") as f:
    f.write(content)
