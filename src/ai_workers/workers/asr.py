"""ASR worker using Qwen3-ASR with Custom FastAPI (merged light + heavy).

Serves both Qwen3-ASR-0.6B (light) and Qwen3-ASR-1.7B (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (BF16, A10G supports natively).
Exposes OpenAI-compatible /v1/audio/transcriptions endpoint.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).

LiteLLM integration:
  model: openai/qwen3-asr-0.6b  (or qwen3-asr-1.7b)
  api_base: https://<modal-url>
"""

import modal

from ai_workers.common.images import transformers_asr_image
from ai_workers.common.volumes import HF_CACHE_DIR, hf_cache_vol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle

# Models served by this single app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-asr-0.6b": {"hf_id": "Qwen/Qwen3-ASR-0.6B"},
    "qwen3-asr-1.7b": {"hf_id": "Qwen/Qwen3-ASR-1.7B"},
}

DEFAULT_MODEL = "qwen3-asr-0.6b"

asr_app = modal.App(
    "ai-workers-qwen3-asr",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@asr_app.cls(
    gpu="A10G",
    image=transformers_asr_image(),
    volumes={HF_CACHE_DIR: hf_cache_vol},
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=5)
class ASRServer:
    """Merged ASR server for Qwen3-ASR-0.6B + 1.7B.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Uses qwen-asr package for inference.
    BF16 precision — A10G supports natively.
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load both ASR models at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from qwen_asr import Qwen3ASRModel

        self.models: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            logger.info("Loading {} ...", hf_id)
            model = Qwen3ASRModel.from_pretrained(
                hf_id,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                cache_dir=HF_CACHE_DIR,
            )
            self.models[name] = model
            logger.info("Loaded {} successfully", name)

    def _load_audio(self, file_bytes: bytes) -> tuple:
        """Convert audio bytes to (numpy_array, sample_rate) tuple for qwen-asr.

        qwen-asr accepts: file path, URL, base64, or (ndarray, sr) tuple.
        Since we receive multipart upload bytes, convert via soundfile.
        """
        import io

        import soundfile as sf

        audio_array, sample_rate = sf.read(io.BytesIO(file_bytes), dtype="float32")
        return (audio_array, sample_rate)

    def _transcribe(self, model_name: str, audio_data, language: str | None = None) -> str:
        """Transcribe audio using the specified model.

        qwen-asr returns a list of result objects with .text and .language attrs.
        Also handles str/dict returns for robustness.
        """
        model = self.models[model_name]
        result = model.transcribe(audio=audio_data, language=language)
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            return result.get("text", "").strip()
        # List of result objects (qwen-asr standard return)
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if hasattr(item, "text"):
                return item.text.strip()
            if isinstance(item, dict):
                return item.get("text", "").strip()
            return str(item).strip()
        return str(result).strip()

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, File, Form, Request, UploadFile
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 ASR (Light + Heavy)")

        class TranscriptionResponse(BaseModel):
            text: str

        class VerboseTranscriptionResponse(BaseModel):
            task: str = "transcribe"
            language: str
            duration: float
            text: str
            segments: list[dict] | None = None

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

        @app.post("/v1/audio/transcriptions")
        async def transcribe(
            file: UploadFile = File(...),
            model: str = Form(DEFAULT_MODEL),
            language: str | None = Form(None),
            response_format: str = Form("json"),
        ):
            """OpenAI-compatible audio transcription endpoint.

            Accepts multipart/form-data with an audio file.
            Supports formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg.
            Routes to light (0.6B) or heavy (1.7B) model via the model field.
            """
            if model not in MODEL_CONFIGS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown model: {model}. Available: {list(MODEL_CONFIGS.keys())}"
                    },
                )

            max_audio_size = 25 * 1024 * 1024  # 25 MB
            chunks: list[bytes] = []
            downloaded = 0
            while chunk := await file.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > max_audio_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": f"Audio file too large ({downloaded} bytes). "
                            f"Maximum allowed: {max_audio_size} bytes (25 MB)."
                        },
                    )
                chunks.append(chunk)
            file_bytes = b"".join(chunks)
            audio_data = self._load_audio(file_bytes)
            text = self._transcribe(model, audio_data, language=language)

            if response_format == "verbose_json":
                return VerboseTranscriptionResponse(
                    language=language or "auto",
                    duration=0.0,
                    text=text,
                    segments=None,
                )

            if response_format == "text":
                from fastapi.responses import PlainTextResponse

                return PlainTextResponse(text)

            # Default: json
            return TranscriptionResponse(text=text)

        return app
