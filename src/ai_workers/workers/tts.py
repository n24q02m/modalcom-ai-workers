"""TTS worker using Qwen3-TTS CustomVoice with Custom FastAPI (merged light + heavy).

Serves both Qwen3-TTS-12Hz-0.6B-CustomVoice (light) and Qwen3-TTS-12Hz-1.7B-CustomVoice (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (BF16, A10G supports natively).
Exposes OpenAI-compatible /v1/audio/speech endpoint.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).

CustomVoice models have 9 preset speakers:
  aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian

Endpoint:
  POST /v1/audio/speech
  Input: JSON {model, input (text), voice (speaker), language, instruct (optional)}
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
DEFAULT_VOICE = "vivian"

# Models served by this single app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-tts-0.6b": {"hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"},
    "qwen3-tts-1.7b": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"},
}

# 9 preset speakers available in CustomVoice models
SUPPORTED_SPEAKERS = [
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian",
]

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
    """Merged TTS server for Qwen3-TTS-12Hz-0.6B + 1.7B CustomVoice.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Uses qwen-tts's Qwen3TTSModel with
    generate_custom_voice() for preset speaker TTS (10 languages).
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
        voice: str = DEFAULT_VOICE,
        language: str = "Auto",
        instruct: str | None = None,
    ) -> tuple:
        """Synthesize speech from text using a preset speaker.

        Returns (wav_numpy_array, sample_rate) tuple.
        Uses generate_custom_voice() with one of 9 preset speakers.
        Optional instruct controls speaking style (e.g., "Very happy").
        """
        model = self.models[model_name]

        kwargs: dict = {
            "text": text,
            "language": language,
            "speaker": voice,
        }

        if instruct:
            kwargs["instruct"] = instruct

        wavs, sr = model.generate_custom_voice(**kwargs)
        # wavs is a list of arrays (one per input text)
        return wavs[0] if isinstance(wavs, list) else wavs, sr

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse, Response
        from pydantic import BaseModel, Field

        app = FastAPI(title="Qwen3 TTS (Light + Heavy)")

        class SpeechRequest(BaseModel):
            model: str = DEFAULT_MODEL
            input: str = Field(max_length=4096)
            voice: str = DEFAULT_VOICE  # OpenAI-compatible: preset speaker name
            language: str = "Auto"
            instruct: str | None = None  # Speaking style instruction (optional)

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
                "speakers": SUPPORTED_SPEAKERS,
            }

        @app.post("/v1/audio/speech")
        async def create_speech(body: SpeechRequest = Body(...)):
            """OpenAI-compatible TTS endpoint.

            Returns WAV audio bytes. Uses preset speakers from CustomVoice models.
            Optional instruct field controls speaking style.
            """
            if body.model not in MODEL_CONFIGS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown model: {body.model}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}"
                    },
                )

            if body.voice.lower() not in SUPPORTED_SPEAKERS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown voice: {body.voice}. Available: {SUPPORTED_SPEAKERS}"
                    },
                )

            import io

            import numpy as np
            import soundfile as sf

            wavs, sr = self._synthesize(
                body.model,
                body.input,
                voice=body.voice,
                language=body.language,
                instruct=body.instruct,
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
