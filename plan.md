1. **Refactor `_do_rerank` in `src/ai_workers/workers/mm_reranker.py`**
   - Import `asyncio` and `from typing import Any` at the top if needed or where appropriate.
   - In `_do_rerank`, collect all unique URLs for images, audio, and videos from the query and all documents.
   - Use `asyncio.gather` and `asyncio.to_thread` to fetch all these URLs concurrently using the class's `_load_image`, `_load_audio`, and `_load_video_frames` static/instance methods. Note that `_load_image` is currently not static (or maybe it is, let's check).
   - Create mapping dictionaries from URLs to their fetched data.

2. **Modify `_score_pair` signature and logic in `src/ai_workers/workers/mm_reranker.py`**
   - Add parameters: `query_image: Any | None = None`, `query_audio: Any | None = None`, `query_video: Any | None = None`.
   - Add parameters: `doc_image: Any | None = None`, `doc_audio: Any | None = None`, `doc_video: Any | None = None`.
   - Use these passed pre-fetched objects to append to the `images` and `audios` lists instead of calling the `_load_*` methods synchronously.

3. **Update `tests/test_workers_mm_reranker.py`**
   - `_score_pair` signature changed, so the mock assertions need to change. Tests will probably check for `query_image_url`, etc., which I'll keep. Let's make sure the tests still verify the URLs correctly. I'll add `Any` imports as well.

4. **Complete Pre-commit Steps**
   - Ensure proper testing, verification, review, and reflection are done by running required linting and testing steps.
