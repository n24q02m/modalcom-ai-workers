"""Multimodal Reranker worker using Custom FastAPI.

Serves gemma4-reranker-v1 (Gemma-4-E4B fine-tuned) on a single A10G container.
Supports text, image, audio, and video modalities in query/document pairs.

Uses AutoModelForImageTextToText with yes/no logit scoring for relevance.
Scoring: sigmoid(yes_logit - no_logit) -> P(relevant), consistent with
reranker.py and vl_reranker.py.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).
"""

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

        Supports text-only and multimodal (image+audio+video+text) pairs.
        Scoring: sigmoid(yes_logit - no_logit) -> P(relevant).
        """
        import torch

        model = self.models[model_name]
        processor = self.processors[model_name]

        # Build user content with optional media
        content_parts = []
        images = []
        audios = []

        # Query media (before text, per Google guidance)
        if query_image_url:
            content_parts.append({"type": "image", "image": query_image_url})
            images.append(self._load_image(query_image_url))
        if query_audio_url:
            content_parts.append({"type": "audio", "audio": query_audio_url})
            audio_data, _sr = self._load_audio(query_audio_url)
            audios.append(audio_data)
        if query_video_url:
            frames = self._load_video_frames(query_video_url)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        content_parts.append({"type": "text", "text": f"<Query>\n{query}\n</Query>"})

        # Document media
        if doc_image_url:
            content_parts.append({"type": "image", "image": doc_image_url})
            images.append(self._load_image(doc_image_url))
        if doc_audio_url:
            content_parts.append({"type": "audio", "audio": doc_audio_url})
            audio_data, _sr = self._load_audio(doc_audio_url)
            audios.append(audio_data)
        if doc_video_url:
            frames = self._load_video_frames(doc_video_url)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        content_parts.append({"type": "text", "text": f"\n<Document>\n{document}\n</Document>"})

        messages = [
            {"role": "system", "content": RERANKER_PREFIX},
            {"role": "user", "content": content_parts},
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build processor kwargs based on available media
        proc_kwargs = {"text": text, "return_tensors": "pt", "padding": True}
        if images:
            proc_kwargs["images"] = images
        if audios:
            proc_kwargs["audios"] = audios

        inputs = processor(**proc_kwargs).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get logits for "yes" and "no" tokens
            yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
            no_id = processor.tokenizer.convert_tokens_to_ids("no")
            yes_logit = logits[yes_id].float()
            no_logit = logits[no_id].float()

            # Sigmoid of (yes - no) gives relevance probability
            score = torch.sigmoid(yes_logit - no_logit).item()

        return score

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
            if body.model not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model: {body.model}. Available: {list(MODEL_CONFIGS.keys())}"
                )

            results = []
            for i, doc_text in enumerate(body.documents):
                doc_image = body.doc_images[i] if body.doc_images is not None else None
                doc_audio = body.doc_audios[i] if body.doc_audios is not None else None
                doc_video = body.doc_videos[i] if body.doc_videos is not None else None

                try:
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
                    )
                except ValueError as media_err:
                    raise ValueError(str(media_err)) from media_err
                except Exception as e:
                    from loguru import logger

                    logger.error("Scoring failed for doc {}: {}", i, e)
                    raise ValueError(f"Failed to score document {i}: {e}") from e

                result = RerankResult(index=i, relevance_score=score)
                if body.return_documents:
                    result.document = RerankResultDocument(text=doc_text)
                results.append(result)

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
