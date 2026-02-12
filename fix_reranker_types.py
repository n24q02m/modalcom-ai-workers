import re

file_path = "src/ai_workers/workers/reranker.py"

with open(file_path, "r") as f:
    content = f.read()

# Add TYPE_CHECKING import
if "from typing import TYPE_CHECKING" not in content:
    content = content.replace("from __future__ import annotations", "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast")

# Update class definitions to add self.tokenizer type hint
# We can't easily add it to class body without knowing where.
# But we can add it to __init__ if it existed, or just ignore the errors?
# No, better to cast or assert.
# Or just define it in load_model with type comment?
# self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = ...

# Let's try adding a type hint in the class body if possible, or just ignore for now as it is dynamic.
# Actually, the error is because 'self.tokenizer' is not defined in __init__, so it's inferred as Unknown/None from somewhere or just not known.
# And inside load_model it is assigned.

# The easiest fix for "possibly-missing-attribute" on "self.tokenizer" which is assigned in "load_model" (Modal lifecycle)
# is to declare it in the class.

# Add declarations to RerankerLightServer
content = content.replace(
    'class RerankerLightServer:\n    """Custom FastAPI reranker server for Qwen3-Reranker-0.6B."""',
    'class RerankerLightServer:\n    """Custom FastAPI reranker server for Qwen3-Reranker-0.6B."""\n\n    if TYPE_CHECKING:\n        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast'
)

# Add declarations to RerankerHeavyServer
content = content.replace(
    'class RerankerHeavyServer:\n    """Custom FastAPI reranker server for Qwen3-Reranker-8B."""',
    'class RerankerHeavyServer:\n    """Custom FastAPI reranker server for Qwen3-Reranker-8B."""\n\n    if TYPE_CHECKING:\n        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast'
)

with open(file_path, "w") as f:
    f.write(content)
