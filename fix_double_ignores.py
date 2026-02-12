import re

def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Replace multiple type ignores with a single one
    # Regex to match "# type: ignore" followed by whitespace and another "# type: ignore" repeated
    content = re.sub(r'(# type: ignore\s*)+', '# type: ignore', content)

    with open(filepath, "w") as f:
        f.write(content)

fix_file("src/ai_workers/workers/reranker.py")
fix_file("src/ai_workers/workers/embedding.py")
fix_file("src/ai_workers/workers/asr.py")
fix_file("src/ai_workers/cli/convert.py")
