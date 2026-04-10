import re

with open("src/ai_workers/workers/mm_reranker.py", "r") as f:
    content = f.read()

# Add imports
if "from typing import Any" not in content:
    content = content.replace("import modal", "import modal\nfrom typing import Any")

if "import asyncio" not in content:
    content = content.replace("import modal", "import asyncio\nimport modal")

# Modify _do_rerank
do_rerank_orig = """        async def _do_rerank(body: MmRerankRequest) -> MmRerankResponse:
            if body.model not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model: {body.model}. "
                    f"Available: {list(MODEL_CONFIGS.keys())}"
                )

            results = []
            for i, doc_text in enumerate(body.documents):"""

do_rerank_new = """        async def _do_rerank(body: MmRerankRequest) -> MmRerankResponse:
            if body.model not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model: {body.model}. "
                    f"Available: {list(MODEL_CONFIGS.keys())}"
                )

            # Pre-fetch media concurrently
            image_urls = set()
            audio_urls = set()
            video_urls = set()

            if body.query_image: image_urls.add(body.query_image)
            if body.query_audio: audio_urls.add(body.query_audio)
            if body.query_video: video_urls.add(body.query_video)

            if body.doc_images:
                for url in body.doc_images:
                    if url: image_urls.add(url)
            if body.doc_audios:
                for url in body.doc_audios:
                    if url: audio_urls.add(url)
            if body.doc_videos:
                for url in body.doc_videos:
                    if url: video_urls.add(url)

            unique_images = list(image_urls)
            unique_audios = list(audio_urls)
            unique_videos = list(video_urls)

            tasks = []
            for url in unique_images:
                tasks.append(asyncio.to_thread(self._load_image, url))
            for url in unique_audios:
                tasks.append(asyncio.to_thread(self._load_audio, url))
            for url in unique_videos:
                tasks.append(asyncio.to_thread(self._load_video_frames, url))

            all_results = await asyncio.gather(*tasks) if tasks else []

            idx = 0
            image_map = {}
            for url in unique_images:
                image_map[url] = all_results[idx]
                idx += 1
            audio_map = {}
            for url in unique_audios:
                audio_map[url] = all_results[idx]
                idx += 1
            video_map = {}
            for url in unique_videos:
                video_map[url] = all_results[idx]
                idx += 1

            results = []
            for i, doc_text in enumerate(body.documents):"""

content = content.replace(do_rerank_orig, do_rerank_new)

with open("src/ai_workers/workers/mm_reranker.py", "w") as f:
    f.write(content)
