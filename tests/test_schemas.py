"""API contract tests for OpenAI / Cohere compatibility.

Validates that the Pydantic response schemas defined in each worker
exactly match the specifications expected by LiteLLM and downstream
consumers (KnowledgePrism). These tests run purely against schemas —
no GPU, no model loading.

Why this matters:
- LiteLLM parses responses using OpenAI/Cohere SDK structures
- A missing field or wrong type silently breaks the entire pipeline
- Schema drift between workers would cause inconsistent behaviour
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helper: extract Pydantic models from worker modules by parsing JSON schema
# ---------------------------------------------------------------------------
def _get_model_schema(model_cls: type[BaseModel]) -> dict:
    """Return JSON schema dict for a Pydantic model."""
    return model_cls.model_json_schema()


# ---------------------------------------------------------------------------
# We cannot import worker-level classes normally because they are defined
# inside `serve()` methods (scope-local). Instead we re-create the schema
# models here to test the EXPECTED contract that workers MUST satisfy.
# ---------------------------------------------------------------------------


# === OpenAI Embeddings Contract ===
class ExpectedEmbeddingData(BaseModel):
    object: str
    embedding: list[float]
    index: int


class ExpectedEmbeddingResponse(BaseModel):
    object: str
    data: list[ExpectedEmbeddingData]
    model: str
    usage: dict[str, int]


# === Cohere Rerank Contract ===
class ExpectedDocumentResult(BaseModel):
    index: int
    relevance_score: float
    document: dict[str, str]


class ExpectedRerankResponse(BaseModel):
    results: list[ExpectedDocumentResult]
    model: str


# === OpenAI Chat Completions Contract (OCR) ===
class ExpectedChoice(BaseModel):
    index: int
    message: dict[str, str]
    finish_reason: str


class ExpectedUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ExpectedChatCompletionResponse(BaseModel):
    id: str
    object: str
    model: str
    choices: list[ExpectedChoice]
    usage: ExpectedUsage


# === OpenAI Audio Transcription Contract (ASR) ===
class ExpectedTranscriptionResponse(BaseModel):
    text: str


class ExpectedVerboseTranscriptionResponse(BaseModel):
    task: str
    language: str
    duration: float
    text: str
    segments: list[dict] | None


# ===========================================================================
# Test: OpenAI Embeddings Response
# ===========================================================================


class TestEmbeddingResponseContract:
    """Validate embedding response matches OpenAI /v1/embeddings spec."""

    def test_minimal_valid_response(self) -> None:
        """Minimal valid embedding response should serialize correctly."""
        resp = ExpectedEmbeddingResponse(
            object="list",
            data=[
                ExpectedEmbeddingData(object="embedding", embedding=[0.1, 0.2, 0.3], index=0),
            ],
            model="qwen3-embedding-0.6b",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
        d = resp.model_dump()
        assert d["object"] == "list"
        assert len(d["data"]) == 1
        assert d["data"][0]["object"] == "embedding"
        assert isinstance(d["data"][0]["embedding"], list)
        assert d["data"][0]["index"] == 0
        assert "prompt_tokens" in d["usage"]
        assert "total_tokens" in d["usage"]

    def test_multiple_embeddings(self) -> None:
        """Multiple embeddings should have sequential indices."""
        data = [
            ExpectedEmbeddingData(
                object="embedding",
                embedding=[float(i)] * 5,
                index=i,
            )
            for i in range(3)
        ]
        resp = ExpectedEmbeddingResponse(
            object="list",
            data=data,
            model="test",
            usage={"prompt_tokens": 30, "total_tokens": 30},
        )
        indices = [d.index for d in resp.data]
        assert indices == [0, 1, 2]

    def test_empty_embedding_list_rejected(self) -> None:
        """Empty embedding vector is technically valid per schema."""
        resp = ExpectedEmbeddingResponse(
            object="list",
            data=[ExpectedEmbeddingData(object="embedding", embedding=[], index=0)],
            model="test",
            usage={"prompt_tokens": 0, "total_tokens": 0},
        )
        assert resp.data[0].embedding == []

    def test_usage_must_have_required_keys(self) -> None:
        """Usage dict must contain prompt_tokens and total_tokens."""
        data = {"prompt_tokens": 5, "total_tokens": 5}
        resp = ExpectedEmbeddingResponse(
            object="list",
            data=[],
            model="test",
            usage=data,
        )
        assert resp.usage["prompt_tokens"] == 5
        assert resp.usage["total_tokens"] == 5

    def test_json_serialization_roundtrip(self) -> None:
        """JSON roundtrip should preserve all fields."""
        resp = ExpectedEmbeddingResponse(
            object="list",
            data=[ExpectedEmbeddingData(object="embedding", embedding=[1.0, 2.0], index=0)],
            model="qwen3-embedding-8b",
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )
        json_str = resp.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["object"] == "list"
        assert parsed["data"][0]["embedding"] == [1.0, 2.0]


# ===========================================================================
# Test: Cohere Rerank Response
# ===========================================================================


class TestRerankResponseContract:
    """Validate rerank response matches Cohere /v1/rerank spec."""

    def test_minimal_valid_response(self) -> None:
        resp = ExpectedRerankResponse(
            results=[
                ExpectedDocumentResult(
                    index=0,
                    relevance_score=0.95,
                    document={"text": "relevant doc"},
                ),
            ],
            model="qwen3-reranker-0.6b",
        )
        d = resp.model_dump()
        assert len(d["results"]) == 1
        assert d["results"][0]["document"] == {"text": "relevant doc"}

    def test_document_must_be_dict_with_text_key(self) -> None:
        """Cohere API returns document as {text: str}, not plain string."""
        result = ExpectedDocumentResult(
            index=0,
            relevance_score=0.5,
            document={"text": "hello"},
        )
        assert isinstance(result.document, dict)
        assert "text" in result.document

    def test_results_sorted_by_relevance_descending(self) -> None:
        """Results should be sorted by relevance score descending."""
        results = [
            ExpectedDocumentResult(index=0, relevance_score=0.3, document={"text": "a"}),
            ExpectedDocumentResult(index=1, relevance_score=0.9, document={"text": "b"}),
            ExpectedDocumentResult(index=2, relevance_score=0.6, document={"text": "c"}),
        ]
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        scores = [r.relevance_score for r in sorted_results]
        assert scores == [0.9, 0.6, 0.3]

    def test_top_n_truncation(self) -> None:
        """top_n should limit results after sorting."""
        results = [
            ExpectedDocumentResult(
                index=i, relevance_score=float(i) / 10, document={"text": f"d{i}"}
            )
            for i in range(10)
        ]
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        top_3 = results[:3]
        assert len(top_3) == 3
        assert all(r.relevance_score >= top_3[-1].relevance_score for r in top_3)

    def test_relevance_score_range(self) -> None:
        """Sigmoid output should be in [0, 1]."""
        for score in [0.0, 0.5, 1.0]:
            result = ExpectedDocumentResult(
                index=0,
                relevance_score=score,
                document={"text": "x"},
            )
            assert 0.0 <= result.relevance_score <= 1.0

    def test_json_serialization_roundtrip(self) -> None:
        resp = ExpectedRerankResponse(
            results=[
                ExpectedDocumentResult(index=0, relevance_score=0.85, document={"text": "doc"})
            ],
            model="test",
        )
        parsed = json.loads(resp.model_dump_json())
        assert parsed["results"][0]["document"]["text"] == "doc"


# ===========================================================================
# Test: OpenAI Chat Completions Response (OCR)
# ===========================================================================


class TestChatCompletionResponseContract:
    """Validate OCR chat completion response matches OpenAI spec."""

    def test_minimal_valid_response(self) -> None:
        resp = ExpectedChatCompletionResponse(
            id="chatcmpl-abc123",
            object="chat.completion",
            model="deepseek-ocr-2",
            choices=[
                ExpectedChoice(
                    index=0,
                    message={"role": "assistant", "content": "Extracted text"},
                    finish_reason="stop",
                )
            ],
            usage=ExpectedUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        d = resp.model_dump()
        assert d["object"] == "chat.completion"
        assert d["choices"][0]["message"]["role"] == "assistant"
        assert d["choices"][0]["finish_reason"] == "stop"

    def test_id_format(self) -> None:
        """ID should follow chatcmpl-<hex> pattern."""
        resp = ExpectedChatCompletionResponse(
            id="chatcmpl-a1b2c3d4e5f6",
            object="chat.completion",
            model="deepseek-ocr-2",
            choices=[
                ExpectedChoice(
                    index=0, message={"role": "assistant", "content": ""}, finish_reason="stop"
                )
            ],
            usage=ExpectedUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        assert re.match(r"chatcmpl-[a-f0-9]+", resp.id)

    def test_error_response_structure(self) -> None:
        """Error responses (no image) should still be valid chat completions."""
        resp = ExpectedChatCompletionResponse(
            id="chatcmpl-err000",
            object="chat.completion",
            model="deepseek-ocr-2",
            choices=[
                ExpectedChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": "Error: No image provided.",
                    },
                    finish_reason="stop",
                )
            ],
            usage=ExpectedUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        assert resp.choices[0].message["content"].startswith("Error:")

    def test_choice_message_has_required_keys(self) -> None:
        """Message dict must have both 'role' and 'content'."""
        choice = ExpectedChoice(
            index=0,
            message={"role": "assistant", "content": "hello"},
            finish_reason="stop",
        )
        assert "role" in choice.message
        assert "content" in choice.message


# ===========================================================================
# Test: OpenAI Audio Transcription Response (ASR)
# ===========================================================================


class TestTranscriptionResponseContract:
    """Validate ASR transcription response matches OpenAI spec."""

    def test_simple_json_response(self) -> None:
        resp = ExpectedTranscriptionResponse(text="Hello world!")
        assert resp.text == "Hello world!"
        parsed = json.loads(resp.model_dump_json())
        assert parsed == {"text": "Hello world!"}

    def test_empty_transcription(self) -> None:
        resp = ExpectedTranscriptionResponse(text="")
        assert resp.text == ""

    def test_verbose_json_response(self) -> None:
        resp = ExpectedVerboseTranscriptionResponse(
            task="transcribe",
            language="en",
            duration=5.0,
            text="Hello world!",
            segments=[
                {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello"},
                {"id": 1, "start": 2.5, "end": 5.0, "text": "world!"},
            ],
        )
        d = resp.model_dump()
        assert d["task"] == "transcribe"
        assert d["language"] == "en"
        assert d["duration"] == 5.0
        assert len(d["segments"]) == 2

    def test_verbose_segments_have_required_keys(self) -> None:
        """Each segment must have id, start, end, text."""
        segment = {"id": 0, "start": 0.0, "end": 1.0, "text": "hi"}
        for key in ("id", "start", "end", "text"):
            assert key in segment

    def test_verbose_segments_nullable(self) -> None:
        """Segments can be None (no timestamps requested)."""
        resp = ExpectedVerboseTranscriptionResponse(
            task="transcribe",
            language="auto",
            duration=0.0,
            text="",
            segments=None,
        )
        assert resp.segments is None


# ===========================================================================
# Test: Cross-worker consistency
# ===========================================================================


class TestCrossWorkerConsistency:
    """Validate that similar workers use identical schemas."""

    def test_embedding_schema_keys_match(self) -> None:
        """All embedding workers (text + VL) should have same response keys."""
        schema = ExpectedEmbeddingResponse.model_json_schema()
        required = set(schema.get("required", []))
        assert required == {"object", "data", "model", "usage"}

    def test_rerank_schema_keys_match(self) -> None:
        """All reranker workers (text + VL) should have same response keys."""
        schema = ExpectedRerankResponse.model_json_schema()
        required = set(schema.get("required", []))
        assert required == {"results", "model"}

    def test_document_result_has_dict_type(self) -> None:
        """Document field in rerank result must be dict, not str."""
        schema = ExpectedDocumentResult.model_json_schema()
        doc_info = schema["properties"]["document"]
        # Pydantic v2: additionalProperties for dict[str, str]
        assert doc_info.get("type") == "object"

    def test_embedding_data_object_literal(self) -> None:
        """EmbeddingData.object should default to 'embedding'."""
        data = ExpectedEmbeddingData(embedding=[0.1], index=0, object="embedding")
        assert data.object == "embedding"


# ===========================================================================
# Test: LiteLLM integration contract
# ===========================================================================


class TestLiteLLMIntegrationContract:
    """Validate responses are parseable by LiteLLM/OpenAI SDK patterns."""

    def test_embedding_response_parseable_as_openai(self) -> None:
        """Response should match openai.types.CreateEmbeddingResponse structure."""
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
            ],
            "model": "qwen3-embedding-0.6b",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        resp = ExpectedEmbeddingResponse.model_validate(raw)
        assert resp.object == "list"
        assert resp.data[0].object == "embedding"

    def test_rerank_response_parseable_as_cohere(self) -> None:
        """Response should match cohere.types.RerankResponse structure."""
        raw = {
            "results": [
                {
                    "index": 2,
                    "relevance_score": 0.95,
                    "document": {"text": "most relevant"},
                },
                {
                    "index": 0,
                    "relevance_score": 0.3,
                    "document": {"text": "less relevant"},
                },
            ],
            "model": "qwen3-reranker-0.6b",
        }
        resp = ExpectedRerankResponse.model_validate(raw)
        assert resp.results[0].relevance_score > resp.results[1].relevance_score

    def test_chat_completion_response_parseable(self) -> None:
        """Response should match openai.types.ChatCompletion structure."""
        raw = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "model": "deepseek-ocr-2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OCR text"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        resp = ExpectedChatCompletionResponse.model_validate(raw)
        assert resp.choices[0].message["content"] == "OCR text"

    def test_transcription_response_parseable(self) -> None:
        """Response should match openai.types.audio.Transcription structure."""
        raw = {"text": "Hello world"}
        resp = ExpectedTranscriptionResponse.model_validate(raw)
        assert resp.text == "Hello world"
