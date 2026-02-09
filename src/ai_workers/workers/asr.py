"""Automatic Speech Recognition (ASR) worker using Whisper Large v3 Turbo.

Exposes OpenAI-compatible /v1/audio/transcriptions endpoint.
Supports 'json', 'verbose_json', and 'text' response formats.
"""

from __future__ import annotations

import modal

from ai_workers.common.images import MODELS_MOUNT_PATH, transformers_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0

r2_mount = get_modal_cloud_bucket_mount()

asr_app = modal.App(
    "ai-workers-whisper-large-v3-turbo",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)

MODEL_NAME = "whisper-large-v3-turbo"


@asr_app.cls(
    gpu="A10G",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class ASRServer:
    """Whisper ASR server."""

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_NAME}"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda",
        )

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, File, Form, Request, UploadFile
        from pydantic import BaseModel

        app = FastAPI(title="Whisper ASR Worker")

        class TranscriptionResponse(BaseModel):
            text: str

        class VerboseTranscriptionResponse(BaseModel):
            language: str
            duration: float
            text: str
            segments: list[dict]

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            from ai_workers.common.auth import verify_api_key

            await verify_api_key(request)
            return await call_next(request)

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_NAME}

        @app.post("/v1/audio/transcriptions")
        async def transcribe(
            file: UploadFile = File(...),
            model: str = Form(MODEL_NAME),
            language: str | None = Form(None),
            prompt: str | None = Form(None),
            response_format: str = Form("json"),
            temperature: float = Form(0.0),
        ):
            import io

            import librosa

            file_bytes = await file.read()
            audio, _ = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)

            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language

            # Whisper pipeline result is a dict with "text" and optionally "chunks" if return_timestamps is True
            result = self.pipe(
                audio,
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=response_format == "verbose_json",
                generate_kwargs=generate_kwargs,
            )

            text = result.get("text", "").strip()  # type: ignore

            if response_format == "verbose_json":
                chunks = result.get("chunks", [])  # type: ignore
                segments = []
                for i, chunk in enumerate(chunks):
                    ts = chunk.get("timestamp", (0, 0))
                    segments.append(
                        {
                            "id": i,
                            "start": ts[0] if ts[0] is not None else 0.0,
                            "end": ts[1] if ts[1] is not None else 0.0,
                            "text": chunk.get("text", ""),
                        }
                    )
                duration = segments[-1]["end"] if segments else 0.0
                return VerboseTranscriptionResponse(
                    language=language or "auto",
                    duration=duration,
                    text=text,
                    segments=segments,
                )

            if response_format == "text":
                from fastapi.responses import PlainTextResponse

                return PlainTextResponse(text)

            # Default: json
            return TranscriptionResponse(text=text)

        return app
