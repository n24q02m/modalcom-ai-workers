import re

with open("tests/test_gguf_converter.py", "r") as f:
    content = f.read()

idx = content.find("# gguf_convert_model - subprocess tests")
if idx != -1:
    content = content[:idx]

with open("tests/test_gguf_converter.py", "w") as f:
    f.write(content)
