"""TTS worker using Qwen3-TTS with Custom FastAPI (merged light + heavy).

Serves both Qwen3-TTS-12Hz-0.6B-Base (light) and Qwen3-TTS-12Hz-1.7B-Base (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (BF16, A10G supports natively).
Exposes OpenAI-compatible /v1/audio/speech endpoint.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).

Base models use voice cloning via generate_voice_clone():
  - With ref_audio + ref_text: clone the reference voice
  - With ref_audio only: x_vector_only_mode (timbre only, lower quality)
  - Without ref_audio: x_vector_only_mode with no ref (default voice)

Endpoint:
  POST /v1/audio/speech
  Input: JSON {model, input (text), language, ref_audio (optional base64/URL), ref_text (optional)}
  Output: audio/wav response
"""

import modal

from ai_workers.common.images import transformers_tts_image
from ai_workers.common.volumes import HF_CACHE_DIR, hf_cache_vol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle
DEFAULT_MODEL = "qwen3-tts-0.6b"

# Models served by this single app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-tts-0.6b": {"hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"},
    "qwen3-tts-1.7b": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"},
}

tts_app = modal.App(
    "ai-workers-tts",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@tts_app.cls(
    gpu="A10G",
    image=transformers_tts_image(),
    volumes={HF_CACHE_DIR: hf_cache_vol},
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=5)
class TTSServer:
    """Merged TTS server for Qwen3-TTS-12Hz-0.6B-Base + 1.7B-Base.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Uses qwen-tts's Qwen3TTSModel for
    voice-cloning TTS (10 languages, 12Hz tokenizer).
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load both TTS models at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from qwen_tts import Qwen3TTSModel

        self.models: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            logger.info("Loading {} ...", hf_id)
            # NOTE: Do NOT pass cache_dir here — qwen-tts sub-model loading
            # (AutoFeatureExtractor for speech_tokenizer) doesn't propagate it.
            # HF_HUB_CACHE env var handles cache resolution consistently.
            model = Qwen3TTSModel.from_pretrained(
                hf_id,
                dtype=torch.bfloat16,
                device_map="cuda:0",
            )
            self.models[name] = model
            logger.info("Loaded {} successfully", name)

    def _synthesize(
        self,
        model_name: str,
        text: str,
        language: str = "Auto",
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> tuple:
        """Synthesize speech from text using voice cloning.

        Returns (wav_numpy_array, sample_rate) tuple.
        Base models always use generate_voice_clone():
          - ref_audio + ref_text: full voice clone
          - ref_audio only: x_vector_only_mode (timbre only)
          - no ref_audio: x_vector_only_mode with no ref
        """
        model = self.models[model_name]

        kwargs: dict = {
            "text": text,
            "language": language,
        }

        if ref_audio and ref_text:
            kwargs["ref_audio"] = ref_audio
            kwargs["ref_text"] = ref_text
        elif ref_audio:
            kwargs["ref_audio"] = ref_audio
            kwargs["x_vector_only_mode"] = True
        else:
            kwargs["x_vector_only_mode"] = True

        wavs, sr = model.generate_voice_clone(**kwargs)
        return wavs, sr

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse, Response
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 TTS (Light + Heavy)")

        class SpeechRequest(BaseModel):
            model: str = DEFAULT_MODEL
            input: str
            language: str = "Auto"
            ref_audio: str | None = None  # base64 data URI or URL
            ref_text: str | None = None  # transcript of reference audio

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

        @app.post("/v1/audio/speech")
        async def create_speech(body: SpeechRequest = Body(...)):
            """OpenAI-compatible TTS endpoint.

            Returns WAV audio bytes. Supports voice cloning via ref_audio
            (base64 data URI or URL) and ref_text.
            """
            if body.model not in MODEL_CONFIGS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown model: {body.model}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}"
                    },
                )

            import io

            import numpy as np
            import soundfile as sf

            wavs, sr = self._synthesize(
                body.model,
                body.input,
                language=body.language,
                ref_audio=body.ref_audio,
                ref_text=body.ref_text,
            )

            # Convert to WAV bytes
            buf = io.BytesIO()
            wav_data = wavs if isinstance(wavs, np.ndarray) else np.array(wavs)
            sf.write(buf, wav_data, sr, format="WAV")
            buf.seek(0)

            return Response(
                content=buf.read(),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=speech.wav"},
            )

        return app
