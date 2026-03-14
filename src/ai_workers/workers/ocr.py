"""OCR worker using DeepSeek-OCR-2 with Custom FastAPI.

DeepSeek-OCR-2 is a 3.34B MoE model specialized for document OCR.
Supports two modes:
  - Free OCR: Extract all text from an image
  - Grounding OCR: Extract text from specific regions

Exposes OpenAI Vision-compatible /v1/chat/completions endpoint.
Single app: deepseek-ocr-2 (A10G, BF16 — model trained in BF16, cannot
convert to FP16 without degradation).

LiteLLM integration:
  model: openai/deepseek-ocr-2
  api_base: https://<modal-url>
"""

from __future__ import annotations

import modal

from ai_workers.common.images import MODELS_MOUNT_PATH, transformers_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0
MODEL_NAME = "deepseek-ocr-2"

r2_mount = get_modal_cloud_bucket_mount()

ocr_app = modal.App(
    "ai-workers-deepseek-ocr-2",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)


@ocr_app.cls(
    gpu="A10G",
    image=transformers_image(flash_attn=True),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=5,
)
class OCRServer:
    """Custom FastAPI OCR server for DeepSeek-OCR-2.

    BF16 precision required — model trained in BF16. Converting to FP16
    causes severe overflow/underflow (BF16 has 8-bit exponent vs FP16's 5-bit).
    A10G supports BF16 natively; T4 does NOT.
    """

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_NAME}"
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            use_safetensors=True,
        )
        self.model.eval()

    def _process_image_content(self, content: list[dict]) -> tuple[str, str | None]:
        """Extract text prompt and image URL from OpenAI content array."""
        text = ""
        image_url = None
        for part in content:
            if part.get("type") == "text":
                text = part.get("text", "")
            elif part.get("type") == "image_url":
                url_data = part.get("image_url", {})
                image_url = url_data.get("url", "")
        return text, image_url

    def _load_image_from_url(self, url: str):
        """Load image from URL or base64 data URI."""
        import base64
        import io

        from PIL import Image

        if url.startswith("data:"):
            # data:image/png;base64,<base64-data>
            _header, b64_data = url.split(",", 1)
            image_bytes = base64.b64decode(b64_data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Regular URL
        import urllib.request

        with urllib.request.urlopen(url) as resp:
            image_bytes = resp.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _run_ocr(self, image, prompt: str = "") -> str:
        """Run OCR on an image with optional prompt.

        DeepSeek-OCR-2 uses model-specific infer() method for OCR tasks.
        """
        import torch

        # Build inputs using the processor
        if prompt:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.model.device, torch.bfloat16)
        else:
            inputs = self.processor(
                images=image,
                return_tensors="pt",
            ).to(self.model.device, torch.bfloat16)

        with torch.no_grad():
            # Try model.infer() first (DeepSeek-OCR-2 specific), fallback to generate
            if hasattr(self.model, "infer"):
                outputs = self.model.infer(
                    images=image,
                    prompts=[prompt] if prompt else None,
                    processor=self.processor,
                )
                if isinstance(outputs, list):
                    return outputs[0] if outputs else ""
                return str(outputs)

            # Fallback: standard generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )
            # Decode only the generated tokens (skip input)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            result = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return result[0] if result else ""

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI(title="DeepSeek OCR v2")

        class ChatMessage(BaseModel):
            role: str
            content: str | list[dict]

        class ChatCompletionRequest(BaseModel):
            model: str = MODEL_NAME
            messages: list[ChatMessage]
            max_tokens: int = 4096
            temperature: float = 0.0

        class Choice(BaseModel):
            index: int = 0
            message: dict[str, str]
            finish_reason: str = "stop"

        class Usage(BaseModel):
            prompt_tokens: int = 0
            completion_tokens: int = 0
            total_tokens: int = 0

        class ChatCompletionResponse(BaseModel):
            id: str
            object: str = "chat.completion"
            model: str
            choices: list[Choice]
            usage: Usage

        from ai_workers.common.auth import auth_middleware

        app.middleware("http")(auth_middleware)

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_NAME}

        @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            import uuid

            # Extract the last user message with image
            text_prompt = ""
            image = None

            for msg in reversed(request.messages):
                if msg.role == "user":
                    if isinstance(msg.content, list):
                        text_prompt, image_url = self._process_image_content(msg.content)
                        if image_url:
                            image = self._load_image_from_url(image_url)
                    elif isinstance(msg.content, str):
                        text_prompt = msg.content
                    break

            if image is None:
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    model=request.model,
                    choices=[
                        Choice(
                            message={
                                "role": "assistant",
                                "content": "Error: No image provided. Please send an image for OCR.",
                            },
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(),
                )

            result = self._run_ocr(image, text_prompt)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                model=request.model,
                choices=[
                    Choice(
                        message={"role": "assistant", "content": result},
                        finish_reason="stop",
                    )
                ],
                usage=Usage(),
            )

        return app
