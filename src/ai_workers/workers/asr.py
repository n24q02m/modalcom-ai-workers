"""ASR worker using Whisper Large v3 with Custom FastAPI.

Whisper Large v3 is a 1.55B parameter speech recognition model.
Exposes OpenAI-compatible /v1/audio/transcriptions endpoint.
Single app: whisper-large-v3 (T4, FP16).

Model downloaded directly from HuggingFace Hub via Xet protocol
at container startup (~1GB/s). No R2 storage needed.

LiteLLM integration:
  model: openai/whisper-large-v3
  api_base: https://<modal-url>
"""

from __future__ import annotations

import modal

from ai_workers.common.images import transformers_audio_image

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0
MODEL_NAME = "whisper-large-v3"
HF_ID = "openai/whisper-large-v3"

asr_app = modal.App(
    "ai-workers-whisper-large-v3",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@asr_app.cls(
    gpu="T4",
    image=transformers_audio_image(),
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=600,
)
@modal.concurrent(max_inputs=5)
class ASRServer:
    """Custom FastAPI ASR server for Whisper Large v3.

    Uses transformers pipeline for robust multilingual speech-to-text.
    Supports chunked long-form audio with automatic language detection.
    """

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from loguru import logger
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        logger.info("Loading {} from HuggingFace Hub...", HF_ID)
        self.processor = AutoProcessor.from_pretrained(HF_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            HF_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info("Loaded {} successfully", MODEL_NAME)

    def _load_audio(self, file_bytes: bytes) -> dict:
        """Load audio bytes into the format expected by the pipeline."""
        import io

        import librosa

        audio, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
        return {"raw": audio, "sampling_rate": sr}

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, File, Form, Request, UploadFile
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI(title="Whisper Large v3")

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
            """OpenAI-compatible audio transcription endpoint.

            Accepts multipart/form-data with an audio file.
            Supports formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg.
            """
            file_bytes = await file.read()
            audio_input = self._load_audio(file_bytes)

            # Build generate_kwargs
            generate_kwargs: dict = {}
            if language:
                generate_kwargs["language"] = language
            if prompt:
                generate_kwargs["initial_prompt"] = prompt
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["do_sample"] = True

            result = self.pipe(
                audio_input,
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=response_format == "verbose_json",
                generate_kwargs=generate_kwargs,
            )

            text = result.get("text", "").strip()

            if response_format == "verbose_json":
                chunks = result.get("chunks", [])
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
