import ast
import sys

def check_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    tree = ast.parse(content)
    has_annotations = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__' and any(alias.name == 'annotations' for alias in node.names):
            has_annotations = True
            break

    if not has_annotations:
        return # No annotations import to check

    # This is a very naive check.
    # In Python 3.13+, 'from __future__ import annotations' is mostly for:
    # 1. Postponed evaluation of annotations (strings instead of types at runtime)
    # 2. Supporting | and list[] in older python versions (but we are on 3.13+)

    print(f"File: {filepath} has annotations import.")

files = [
    "src/ai_workers/common/config.py",
    "src/ai_workers/common/volumes.py",
    "src/ai_workers/common/auth.py",
    "src/ai_workers/common/utils.py",
    "src/ai_workers/common/images.py",
    "src/ai_workers/workers/onnx_converter.py",
    "src/ai_workers/workers/gguf_converter.py",
    "src/ai_workers/cli/deploy.py",
    "src/ai_workers/cli/__main__.py",
    "src/ai_workers/cli/onnx_convert.py",
    "src/ai_workers/cli/gguf_convert.py",
]

for f in files:
    check_file(f)
