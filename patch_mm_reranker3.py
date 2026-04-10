import re

with open("src/ai_workers/workers/mm_reranker.py", "r") as f:
    content = f.read()

score_pair_def_orig = """    def _score_pair(
        self,
        model_name: str,
        query: str,
        document: str,
        query_image_url: str | None = None,
        query_audio_url: str | None = None,
        query_video_url: str | None = None,
        doc_image_url: str | None = None,
        doc_audio_url: str | None = None,
        doc_video_url: str | None = None,
    ) -> float:"""

score_pair_def_new = """    def _score_pair(
        self,
        model_name: str,
        query: str,
        document: str,
        query_image_url: str | None = None,
        query_audio_url: str | None = None,
        query_video_url: str | None = None,
        doc_image_url: str | None = None,
        doc_audio_url: str | None = None,
        doc_video_url: str | None = None,
        query_image: Any | None = None,
        query_audio: Any | None = None,
        query_video: Any | None = None,
        doc_image: Any | None = None,
        doc_audio: Any | None = None,
        doc_video: Any | None = None,
    ) -> float:"""

content = content.replace(score_pair_def_orig, score_pair_def_new)

score_pair_logic_orig = """        # Query media (before text, per Google guidance)
        if query_image_url:
            content_parts.append({"type": "image", "image": query_image_url})
            images.append(self._load_image(query_image_url))
        if query_audio_url:
            content_parts.append({"type": "audio", "audio": query_audio_url})
            audio_data, sr = self._load_audio(query_audio_url)
            audios.append(audio_data)
        if query_video_url:
            frames = self._load_video_frames(query_video_url)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        content_parts.append({"type": "text", "text": f"<Query>\\n{query}\\n</Query>"})

        # Document media
        if doc_image_url:
            content_parts.append({"type": "image", "image": doc_image_url})
            images.append(self._load_image(doc_image_url))
        if doc_audio_url:
            content_parts.append({"type": "audio", "audio": doc_audio_url})
            audio_data, sr = self._load_audio(doc_audio_url)
            audios.append(audio_data)
        if doc_video_url:
            frames = self._load_video_frames(doc_video_url)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)"""

score_pair_logic_new = """        # Query media (before text, per Google guidance)
        if query_image_url:
            content_parts.append({"type": "image", "image": query_image_url})
            images.append(query_image if query_image is not None else self._load_image(query_image_url))
        if query_audio_url:
            content_parts.append({"type": "audio", "audio": query_audio_url})
            if query_audio is not None:
                audio_data, _sr = query_audio
            else:
                audio_data, _sr = self._load_audio(query_audio_url)
            audios.append(audio_data)
        if query_video_url:
            frames = query_video if query_video is not None else self._load_video_frames(query_video_url)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        content_parts.append({"type": "text", "text": f"<Query>\\n{query}\\n</Query>"})

        # Document media
        if doc_image_url:
            content_parts.append({"type": "image", "image": doc_image_url})
            images.append(doc_image if doc_image is not None else self._load_image(doc_image_url))
        if doc_audio_url:
            content_parts.append({"type": "audio", "audio": doc_audio_url})
            if doc_audio is not None:
                audio_data, _sr = doc_audio
            else:
                audio_data, _sr = self._load_audio(doc_audio_url)
            audios.append(audio_data)
        if doc_video_url:
            frames = doc_video if doc_video is not None else self._load_video_frames(doc_video_url)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)"""

content = content.replace(score_pair_logic_orig, score_pair_logic_new)

with open("src/ai_workers/workers/mm_reranker.py", "w") as f:
    f.write(content)
