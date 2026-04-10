import re

def update_file(filename, replacements):
    with open(filename, "r") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filename, "w") as f:
        f.write(content)

update_file("tests/test_config.py", [
    ("assert len(MODEL_REGISTRY) == 11", "assert len(MODEL_REGISTRY) == 12"),
    ("We expect 11 models", "We expect 12 models"),
    ("assert len(models) == 11", "assert len(models) == 12"),
    ("assert len(light) + len(heavy) == 11", "assert len(light) + len(heavy) == 12"),
    ('"qwen3-asr-1.7b": "ai-workers-qwen3-asr",', '"qwen3-asr-1.7b": "ai-workers-qwen3-asr",\n            "gemma4-reranker-v1": "ai-workers-mm-reranker",'),
])

update_file("tests/test_deploy.py", [
    ("assert len(modules) == 7", "assert len(modules) == 8"),
    ("assert len(targets) == 7", "assert len(targets) == 8"),
    ("7 unique worker modules for 11 models.", "8 unique worker modules for 12 models."),
    ("7 unique (module, app_var) pairs for 11 models", "8 unique (module, app_var) pairs for 12 models"),
    ("Plus OCR = 7 total", "Plus OCR and MM Reranker = 8 total"),
])

update_file("tests/test_deploy_extra.py", [
    ("assert mock_subprocess.run.call_count == 7", "assert mock_subprocess.run.call_count == 8"),
    ("There are 7 unique deploy targets", "There are 8 unique deploy targets"),
    ("(embedding, reranker, vl_embedding, vl_reranker, ocr, tts, asr)", "(embedding, reranker, vl_embedding, vl_reranker, ocr, tts, asr, mm_reranker)"),
])
