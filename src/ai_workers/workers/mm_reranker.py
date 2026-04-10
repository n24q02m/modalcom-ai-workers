"""Multimodal Reranker worker using Custom FastAPI.

Serves gemma4-reranker-v1 (Gemma-4-E4B fine-tuned) on a single A10G container.
Supports text, image, audio, and video modalities in query/document pairs.

Uses AutoModelForImageTextToText with yes/no logit scoring for relevance.
Scoring: sigmoid(yes_logit - no_logit) -> P(relevant), consistent with
reranker.py and vl_reranker.py.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).
"""

from typing import Any

import modal

from ai_workers.common.images import transformers_mm_reranker_image
from ai_workers.common.volumes import HF_CACHE_DIR, hf_cache_vol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle

# Model served by this app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "gemma4-reranker-v1": {"hf_id": "n24q02m/gemma4-e4b-reranker-v1"},
}

# System prompt for relevance judgment
RERANKER_PREFIX = 'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".'

# Media constraints
MAX_AUDIO_DURATION_S = 30.0
MAX_VIDEO_DURATION_S = 60.0
MAX_VIDEO_FRAMES = 8
MEDIA_DOWNLOAD_TIMEOUT_S = 30

mm_reranker_app = modal.App(
    "ai-workers-mm-reranker",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@mm_reranker_app.cls(
    gpu="A10G",
    image=transformers_mm_reranker_image(),
    volumes={HF_CACHE_DIR: hf_cache_vol},
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=1800,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class MmRerankerServer:
    """Multimodal reranker server for Gemma-4-E4B.

    Supports text, image, audio, and video inputs.
    Uses yes/no logit scoring with sigmoid for relevance probability.
    """

    @modal.enter(snap=True)
    def load_model(self) -> None:
        """Load model at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.models: dict[str, object] = {}
        self.processors: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            logger.info("Loading {} ...", hf_id)
            processor = AutoProcessor.from_pretrained(
                hf_id,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
            )
            model.eval()
            self.models[name] = model
            self.processors[name] = processor
            logger.info("Loaded {} successfully", name)

    # ---------------------------------------------------------------------------
    # Media loading helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _load_image(url: str):
        """Load a PIL Image from URL. Timeout 30s."""
        import requests as http_requests
        from PIL import Image

        resp = http_requests.get(url, stream=True, timeout=MEDIA_DOWNLOAD_TIMEOUT_S)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")

    @staticmethod
    def _load_audio(url: str) -> tuple:
        """Load audio from URL as numpy array @ original sample rate.

        Returns (waveform_np, sample_rate).
        Raises ValueError if duration > MAX_AUDIO_DURATION_S.
        """
        import io

        import requests as http_requests
        import soundfile as sf

        resp = http_requests.get(url, timeout=MEDIA_DOWNLOAD_TIMEOUT_S)
        resp.raise_for_status()

        # soundfile needs a seekable file-like object
        buf = io.BytesIO(resp.content)
        data, sr = sf.read(buf, dtype="float32")

        duration = len(data) / sr if data.ndim == 1 else data.shape[0] / sr
        if duration > MAX_AUDIO_DURATION_S:
            raise ValueError(
                f"Audio duration {duration:.1f}s exceeds maximum {MAX_AUDIO_DURATION_S}s"
            )

        return data, sr

    @staticmethod
    def _load_video_frames(url: str) -> list:
        """Download video and extract uniform frames.

        Returns list of PIL.Image (RGB), max MAX_VIDEO_FRAMES.
        Raises ValueError if duration > MAX_VIDEO_DURATION_S.
        """
        import tempfile

        import av
        import numpy as np
        import requests as http_requests

        resp = http_requests.get(url, timeout=MEDIA_DOWNLOAD_TIMEOUT_S)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(resp.content)
            tmp.flush()

            container = av.open(tmp.name)
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)

            if duration > MAX_VIDEO_DURATION_S:
                container.close()
                raise ValueError(
                    f"Video duration {duration:.1f}s exceeds maximum {MAX_VIDEO_DURATION_S}s"
                )

            n_frames = min(MAX_VIDEO_FRAMES, max(1, int(duration)))  # ~1 fps
            timestamps = np.linspace(0, duration, n_frames, endpoint=False)

            frames = []
            for ts in timestamps:
                container.seek(int(ts / stream.time_base), stream=stream)
                for frame in container.decode(video=0):
                    frames.append(frame.to_image())
                    break

            container.close()

        return frames[:MAX_VIDEO_FRAMES]

    # ---------------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------------

    def _score_batch(
        self,
        model_name: str,
        query: str,
        documents: list[str],
        query_image: Any | None = None,
        query_audio: Any | None = None,
        query_video_frames: list[Any] | None = None,
        doc_images: list[Any | None] | None = None,
        doc_audios: list[Any | None] | None = None,
        doc_video_frames: list[list[Any] | None] | None = None,
    ) -> list[float]:
        """Score a batch of query-document pairs using yes/no logit comparison.

        Supports text-only and multimodal (image+audio+video+text) pairs.
        Scoring: sigmoid(yes_logit - no_logit) -> P(relevant).
        """
        import torch

        model = self.models[model_name]
        processor = self.processors[model_name]

        all_scores = []
        batch_size = 4  # Small batch size for A10G memory safety with multimodal models

        for chunk_idx in range(0, len(documents), batch_size):
            chunk_docs = documents[chunk_idx : chunk_idx + batch_size]
            chunk_doc_images = (
                doc_images[chunk_idx : chunk_idx + batch_size] if doc_images else None
            )
            chunk_doc_audios = (
                doc_audios[chunk_idx : chunk_idx + batch_size] if doc_audios else None
            )
            chunk_doc_videos = (
                doc_video_frames[chunk_idx : chunk_idx + batch_size] if doc_video_frames else None
            )

            chunk_messages = []
            chunk_all_images = []
            chunk_all_audios = []

            for i, doc_text in enumerate(chunk_docs):
                content_parts = []
                images = []
                audios = []

                # Query media (before text, per Google guidance)
                if query_image:
                    content_parts.append({"type": "image", "image": query_image})
                    images.append(query_image)
                if query_audio:
                    content_parts.append({"type": "audio", "audio": query_audio})
                    audios.append(query_audio)
                if query_video_frames:
                    for frame in query_video_frames:
                        content_parts.append({"type": "image", "image": frame})
                        images.append(frame)

                content_parts.append({"type": "text", "text": f"<Query>\n{query}\n</Query>"})

                # Document media
                if chunk_doc_images and chunk_doc_images[i]:
                    img = chunk_doc_images[i]
                    content_parts.append({"type": "image", "image": img})
                    images.append(img)
                if chunk_doc_audios and chunk_doc_audios[i]:
                    aud = chunk_doc_audios[i]
                    content_parts.append({"type": "audio", "audio": aud})
                    audios.append(aud)
                if chunk_doc_videos and chunk_doc_videos[i]:
                    for frame in chunk_doc_videos[i]:
                        content_parts.append({"type": "image", "image": frame})
                        images.append(frame)

                content_parts.append(
                    {"type": "text", "text": f"\n<Document>\n{doc_text}\n</Document>"}
                )

                messages = [
                    {"role": "system", "content": RERANKER_PREFIX},
                    {"role": "user", "content": content_parts},
                ]
                chunk_messages.append(messages)
                chunk_all_images.append(images)
                chunk_all_audios.append(audios)

            texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in chunk_messages
            ]

            # Build processor kwargs based on available media
            proc_kwargs = {"text": texts, "return_tensors": "pt", "padding": True}

            # Flatten all images/audios for the processor if any are present
            if any(chunk_all_images):
                proc_kwargs["images"] = [img for sub in chunk_all_images for img in sub]
            if any(chunk_all_audios):
                proc_kwargs["audios"] = [aud for sub in chunk_all_audios for aud in sub]

            inputs = processor(**proc_kwargs).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Last token logits for each sequence in batch
                # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
                logits = outputs.logits[:, -1, :]

                # Get logits for "yes" and "no" tokens
                yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
                no_id = processor.tokenizer.convert_tokens_to_ids("no")
                yes_logits = logits[:, yes_id].float()
                no_logits = logits[:, no_id].float()

                # Sigmoid of (yes - no) gives relevance probability
                scores = torch.sigmoid(yes_logits - no_logits).cpu().tolist()
                all_scores.extend(scores)

        return all_scores

    def _score_pair(
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
    ) -> float:
        """Score a single query-document pair using yes/no logit comparison.

        Thin wrapper around `_score_batch` for consistency.
        """
        # Note: This is now slower than calling _score_batch directly because it
        # doesn't benefit from concurrent fetching if used in a loop,
        # but it maintains the legacy interface for tests.
        q_img = self._load_image(query_image_url) if query_image_url else None
        q_aud = self._load_audio(query_audio_url)[0] if query_audio_url else None
        q_vid = self._load_video_frames(query_video_url) if query_video_url else None

        d_img = self._load_image(doc_image_url) if doc_image_url else None
        d_aud = self._load_audio(doc_audio_url)[0] if doc_audio_url else None
        d_vid = self._load_video_frames(doc_video_url) if doc_video_url else None

        scores = self._score_batch(
            model_name,
            query,
            [document],
            query_image=q_img,
            query_audio=q_aud,
            query_video_frames=q_vid,
            doc_images=[d_img],
            doc_audios=[d_aud],
            doc_video_frames=[d_vid],
        )
        return scores[0]

    # ---------------------------------------------------------------------------
    # FastAPI app
    # ---------------------------------------------------------------------------

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, model_validator

        app = FastAPI(title="Gemma-4 Multimodal Reranker")

        class MmRerankRequest(BaseModel):
            model: str = "gemma4-reranker-v1"
            query: str
            query_image: str | None = None
            query_audio: str | None = None
            query_video: str | None = None
            documents: list[str]
            doc_images: list[str | None] | None = None
            doc_audios: list[str | None] | None = None
            doc_videos: list[str | None] | None = None
            top_n: int | None = None
            return_documents: bool = False

            @model_validator(mode="after")
            def check_media_lengths(self):
                for field_name in ("doc_images", "doc_audios", "doc_videos"):
                    field_val = getattr(self, field_name)
                    if field_val is not None and len(field_val) != len(self.documents):
                        raise ValueError(
                            f"{field_name} length ({len(field_val)}) "
                            f"must match documents length ({len(self.documents)})"
                        )
                return self

        class RerankResultDocument(BaseModel):
            text: str

        class RerankResult(BaseModel):
            index: int
            relevance_score: float
            document: RerankResultDocument | None = None

        class MmRerankResponse(BaseModel):
            model: str
            results: list[RerankResult]

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            from fastapi import HTTPException as _HTTPException

            from ai_workers.common.auth import verify_api_key

            try:
                await verify_api_key(request)
            except _HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                )
            return await call_next(request)

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "models": list(MODEL_CONFIGS.keys()),
            }

        async def _do_rerank(body: MmRerankRequest) -> MmRerankResponse:
            import asyncio

            if body.model not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model: {body.model}. Available: {list(MODEL_CONFIGS.keys())}"
                )

            # Pre-fetch all media concurrently to avoid sequential network latency
            urls_to_fetch = set()
            if body.query_image:
                urls_to_fetch.add(body.query_image)
            if body.query_audio:
                urls_to_fetch.add(body.query_audio)
            if body.query_video:
                urls_to_fetch.add(body.query_video)

            if body.doc_images:
                urls_to_fetch.update(u for u in body.doc_images if u)
            if body.doc_audios:
                urls_to_fetch.update(u for u in body.doc_audios if u)
            if body.doc_videos:
                urls_to_fetch.update(u for u in body.doc_videos if u)

            # Create mapping for each media type

            # Remaining URLs that didn't match extension - try to guess or just fetch all
            # For simplicity, we'll just fetch them based on where they appear in the request

            # Better approach: fetch based on usage
            async def safe_fetch(fetch_func, url):
                try:
                    return await asyncio.to_thread(fetch_func, url)
                except Exception as e:
                    return e

            # Gather all required unique fetches
            unique_image_urls = set()
            if body.query_image:
                unique_image_urls.add(body.query_image)
            if body.doc_images:
                unique_image_urls.update(u for u in body.doc_images if u)

            unique_audio_urls = set()
            if body.query_audio:
                unique_audio_urls.add(body.query_audio)
            if body.doc_audios:
                unique_audio_urls.update(u for u in body.doc_audios if u)

            unique_video_urls = set()
            if body.query_video:
                unique_video_urls.add(body.query_video)
            if body.doc_videos:
                unique_video_urls.update(u for u in body.doc_videos if u)

            image_results = await asyncio.gather(
                *(safe_fetch(self._load_image, u) for u in unique_image_urls)
            )
            image_map = dict(zip(unique_image_urls, image_results, strict=True))

            audio_results = await asyncio.gather(
                *(safe_fetch(self._load_audio, u) for u in unique_audio_urls)
            )
            # _load_audio returns (data, sr), we only need data[0]
            audio_map = {
                u: (res[0] if not isinstance(res, Exception) else res)
                for u, res in zip(unique_audio_urls, audio_results, strict=True)
            }

            video_results = await asyncio.gather(
                *(safe_fetch(self._load_video_frames, u) for u in unique_video_urls)
            )
            video_map = dict(zip(unique_video_urls, video_results, strict=True))

            # Helper to get result or raise
            def get_media(u, mapping):
                if u is None:
                    return None
                res = mapping.get(u)
                if isinstance(res, Exception):
                    raise res
                return res

            try:
                q_img = get_media(body.query_image, image_map)
                q_aud = get_media(body.query_audio, audio_map)
                q_vid = get_media(body.query_video, video_map)

                d_imgs = [
                    get_media(u, image_map)
                    for u in (body.doc_images or [None] * len(body.documents))
                ]
                d_auds = [
                    get_media(u, audio_map)
                    for u in (body.doc_audios or [None] * len(body.documents))
                ]
                d_vids = [
                    get_media(u, video_map)
                    for u in (body.doc_videos or [None] * len(body.documents))
                ]
            except ValueError as e:
                raise ValueError(str(e)) from e
            except Exception as e:
                raise ValueError(f"Media loading failed: {e}") from e

            # Score all documents in a batched call
            try:
                scores = self._score_batch(
                    body.model,
                    body.query,
                    body.documents,
                    query_image=q_img,
                    query_audio=q_aud,
                    query_video_frames=q_vid,
                    doc_images=d_imgs,
                    doc_audios=d_auds,
                    doc_video_frames=d_vids,
                )
            except Exception as e:
                from loguru import logger

                logger.error("Batched scoring failed for model {}: {}", body.model, e)
                raise ValueError(f"Scoring failed: {e}") from e

            results = [
                RerankResult(index=i, relevance_score=score) for i, score in enumerate(scores)
            ]

            if body.return_documents:
                for i, doc_text in enumerate(body.documents):
                    results[i].document = RerankResultDocument(text=doc_text)

            # Sort by relevance score descending
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Apply top_n if specified
            if body.top_n is not None:
                results = results[: body.top_n]

            return MmRerankResponse(model=body.model, results=results)

        @app.post("/v1/rerank", response_model=MmRerankResponse)
        async def rerank_v1(body: MmRerankRequest = Body(...)):
            try:
                return await _do_rerank(body)
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})

        @app.post("/v2/rerank", response_model=MmRerankResponse)
        async def rerank_v2(body: MmRerankRequest = Body(...)):
            try:
                return await _do_rerank(body)
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})

        return app
