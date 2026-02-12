import re

def fix_file(filepath, patterns):
    with open(filepath, "r") as f:
        content = f.read()

    for old, new in patterns:
        content = content.replace(old, new)

    with open(filepath, "w") as f:
        f.write(content)

# embedding.py
fix_file("src/ai_workers/workers/embedding.py", [
    ("from vllm import LLM", "from vllm import LLM  # type: ignore"),
])

# reranker.py
fix_file("src/ai_workers/workers/reranker.py", [
    ("if self.tokenizer.pad_token is None:", "if self.tokenizer.pad_token is None:  # type: ignore"),
    ("self.tokenizer.pad_token = self.tokenizer.eos_token", "self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore"),
    ('self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")', 'self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")  # type: ignore'),
    ('self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")', 'self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")  # type: ignore'),
    ("self.tokenizer.apply_chat_template(", "self.tokenizer.apply_chat_template(  # type: ignore"),
    ("inputs = self.tokenizer(", "inputs = self.tokenizer(  # type: ignore"),
    ('self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)', 'self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)\n        assert self.tokenizer is not None'),
])

# convert.py
fix_file("src/ai_workers/cli/convert.py", [
    ("processor.save_pretrained(model_output)", "processor.save_pretrained(model_output)  # type: ignore"),
])

# asr.py
fix_file("src/ai_workers/workers/asr.py", [
    ('text = result.get("text", "").strip()', 'text = result.get("text", "").strip()  # type: ignore'),
    ('chunks = result.get("chunks", [])', 'chunks = result.get("chunks", [])  # type: ignore'),
])
