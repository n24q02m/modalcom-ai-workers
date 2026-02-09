# Learnings

- **Modal Worker Refactoring**: Refactoring identical Modal worker classes into a base class with `@modal.enter` and `@modal.asgi_app` works well and reduces code duplication.
- **Pydantic & Future Annotations**: Moving Pydantic models to module level is crucial for `from __future__ import annotations` compatibility with FastAPI/Pydantic, especially when used inside decorated functions.
- **Transformers Batching**: Setting `pad_token` explicitly is necessary for batched inference with `transformers` if the model doesn't have one by default. Using `attention_mask.sum(dim=1) - 1` is a reliable way to find the last token index in a padded batch for CausalLM.
- **CI Dependency Management**: When using , use  if your project defines optional dependencies (extras) but no dependency groups, as  will not install extras.
