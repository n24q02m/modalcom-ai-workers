import re

with open("src/ai_workers/workers/mm_reranker.py", "r") as f:
    content = f.read()

score_pair_orig = """                try:
                    score = self._score_pair(
                        body.model,
                        body.query,
                        doc_text,
                        query_image_url=body.query_image,
                        query_audio_url=body.query_audio,
                        query_video_url=body.query_video,
                        doc_image_url=doc_image,
                        doc_audio_url=doc_audio,
                        doc_video_url=doc_video,
                    )"""

score_pair_new = """                try:
                    score = self._score_pair(
                        body.model,
                        body.query,
                        doc_text,
                        query_image_url=body.query_image,
                        query_audio_url=body.query_audio,
                        query_video_url=body.query_video,
                        doc_image_url=doc_image,
                        doc_audio_url=doc_audio,
                        doc_video_url=doc_video,
                        query_image=image_map.get(body.query_image) if body.query_image else None,
                        query_audio=audio_map.get(body.query_audio) if body.query_audio else None,
                        query_video=video_map.get(body.query_video) if body.query_video else None,
                        doc_image=image_map.get(doc_image) if doc_image else None,
                        doc_audio=audio_map.get(doc_audio) if doc_audio else None,
                        doc_video=video_map.get(doc_video) if doc_video else None,
                    )"""

content = content.replace(score_pair_orig, score_pair_new)

with open("src/ai_workers/workers/mm_reranker.py", "w") as f:
    f.write(content)
