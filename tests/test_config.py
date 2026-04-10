"""Tests for model registry configuration.

Validates that all model configs are consistent and complete.
"""

from __future__ import annotations

import pytest

from ai_workers.common.config import (
    GPU,
    MODEL_REGISTRY,
    ModelClassType,
    ModelConfig,
    Precision,
    ServingEngine,
    Task,
    Tier,
    get_model,
    get_model_class,
    get_torch_dtype,
    list_models,
)


class TestModelRegistry:
    """Test the model registry is properly configured."""

    def test_registry_not_empty(self) -> None:
        assert len(MODEL_REGISTRY) > 0

    def test_expected_model_count(self) -> None:
        """We expect 11 models in the registry."""
        assert len(MODEL_REGISTRY) == 12

    @pytest.mark.parametrize(
        "name",
        [
            "qwen3-embedding-0.6b",
            "qwen3-embedding-8b",
            "qwen3-reranker-8b",
            "qwen3-vl-embedding-2b",
            "qwen3-vl-embedding-8b",
            "qwen3-vl-reranker-8b",
            "deepseek-ocr-2",
            "qwen3-tts-0.6b",
            "qwen3-tts-1.7b",
            "qwen3-asr-0.6b",
            "qwen3-asr-1.7b",
        ],
    )
    def test_model_exists(self, name: str) -> None:
        model = get_model(name)
        assert isinstance(model, ModelConfig)
        assert model.name == name

    def test_get_model_invalid_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get_model("nonexistent-model")

    def test_all_models_have_required_fields(self) -> None:
        for name, config in MODEL_REGISTRY.items():
            assert config.name == name
            assert config.hf_id, f"{name}: missing hf_id"
            assert config.task in Task, f"{name}: invalid task"
            assert config.tier in Tier, f"{name}: invalid tier"
            assert config.precision in Precision, f"{name}: invalid precision"
            assert config.gpu in GPU, f"{name}: invalid gpu"
            assert config.serving_engine in ServingEngine, f"{name}: invalid serving_engine"
            assert config.worker_module, f"{name}: missing worker_module"

    def test_modal_app_name_correct(self) -> None:
        """modal_app_name should be correctly set (explicit or auto-derived)."""
        # Merged apps have explicit modal_app_name (shared by light+heavy)
        expected_names = {
            "qwen3-embedding-0.6b": "ai-workers-embedding",
            "qwen3-embedding-8b": "ai-workers-embedding",
            "qwen3-reranker-8b": "ai-workers-reranker",
            "qwen3-vl-embedding-2b": "ai-workers-vl-embedding",
            "qwen3-vl-embedding-8b": "ai-workers-vl-embedding",
            "qwen3-vl-reranker-8b": "ai-workers-vl-reranker",
            "deepseek-ocr-2": "ai-workers-deepseek-ocr-2",
            "qwen3-tts-0.6b": "ai-workers-tts",
            "qwen3-tts-1.7b": "ai-workers-tts",
            "qwen3-asr-0.6b": "ai-workers-qwen3-asr",
            "qwen3-asr-1.7b": "ai-workers-qwen3-asr",
            "gemma4-reranker-v1": "ai-workers-mm-reranker",
        }
        for config in MODEL_REGISTRY.values():
            assert config.modal_app_name == expected_names[config.name], (
                f"{config.name}: expected {expected_names[config.name]}, got {config.modal_app_name}"
            )

    def test_all_models_have_modal_app_var(self) -> None:
        """All models must have modal_app_var set for deploy targeting."""
        for name, config in MODEL_REGISTRY.items():
            assert config.modal_app_var, f"{name}: missing modal_app_var"

    def test_bf16_only_on_a10g(self) -> None:
        """BF16 models must use A10G+ (T4 doesn't support BF16)."""
        for config in MODEL_REGISTRY.values():
            if config.precision == Precision.BF16:
                assert config.gpu == GPU.A10G, (
                    f"{config.name}: BF16 requires A10G+, got {config.gpu}"
                )

    def test_deepseek_ocr_is_bf16(self) -> None:
        """DeepSeek-OCR-2 must be BF16 (trained in BF16, FP16 causes degradation)."""
        ocr = get_model("deepseek-ocr-2")
        assert ocr.precision == Precision.BF16

    def test_qwen3_tts_is_bf16(self) -> None:
        """Qwen3-TTS models must be BF16."""
        for name in ("qwen3-tts-0.6b", "qwen3-tts-1.7b"):
            model = get_model(name)
            assert model.precision == Precision.BF16

    def test_qwen3_asr_is_bf16(self) -> None:
        """Qwen3-ASR models must be BF16."""
        for name in ("qwen3-asr-0.6b", "qwen3-asr-1.7b"):
            model = get_model(name)
            assert model.precision == Precision.BF16

    def test_all_models_use_custom_fastapi(self) -> None:
        """All models should use CUSTOM_FASTAPI serving engine (no vLLM)."""
        for config in MODEL_REGISTRY.values():
            assert config.serving_engine == ServingEngine.CUSTOM_FASTAPI, (
                f"{config.name}: expected CUSTOM_FASTAPI, got {config.serving_engine}"
            )


class TestListModels:
    """Test list_models filter functionality."""

    def test_list_all(self) -> None:
        models = list_models()
        assert len(models) == 12

    def test_filter_by_task(self) -> None:
        embeddings = list_models(task=Task.EMBEDDING)
        assert len(embeddings) == 2
        assert all(m.task == Task.EMBEDDING for m in embeddings)

    def test_filter_by_tier(self) -> None:
        light = list_models(tier=Tier.LIGHT)
        assert all(m.tier == Tier.LIGHT for m in light)
        heavy = list_models(tier=Tier.HEAVY)
        assert all(m.tier == Tier.HEAVY for m in heavy)
        assert len(light) + len(heavy) == 12

    def test_filter_by_task_and_tier(self) -> None:
        light_embed = list_models(task=Task.EMBEDDING, tier=Tier.LIGHT)
        assert len(light_embed) == 1
        assert light_embed[0].name == "qwen3-embedding-0.6b"


class TestHelpers:
    """Test helper functions."""

    def test_get_torch_dtype_fp16(self) -> None:
        import torch

        assert get_torch_dtype(Precision.FP16) == torch.float16

    def test_get_torch_dtype_bf16(self) -> None:
        import torch

        assert get_torch_dtype(Precision.BF16) == torch.bfloat16

    def test_get_model_class_auto(self) -> None:
        from transformers import AutoModel

        assert get_model_class(ModelClassType.AUTO_MODEL) is AutoModel

    def test_get_model_class_causal(self) -> None:
        from transformers import AutoModelForCausalLM

        assert get_model_class(ModelClassType.CAUSAL_LM) is AutoModelForCausalLM

    def test_get_model_class_image_text_to_text(self) -> None:
        from transformers import AutoModelForImageTextToText

        assert get_model_class(ModelClassType.IMAGE_TEXT_TO_TEXT) is AutoModelForImageTextToText

    def test_get_model_class_seq2seq(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq

        assert get_model_class(ModelClassType.SEQ2SEQ) is AutoModelForSpeechSeq2Seq


class TestWorkerModulePaths:
    """Validate worker_module paths reference real modules."""

    def test_unique_worker_modules_per_task(self) -> None:
        """Each task type should map to one worker module."""
        task_modules: dict[Task, set[str]] = {}
        for config in MODEL_REGISTRY.values():
            task_modules.setdefault(config.task, set()).add(config.worker_module)

        for task, modules in task_modules.items():
            assert len(modules) == 1, f"Task {task.value} has multiple worker modules: {modules}"

    @pytest.mark.parametrize(
        ("task", "expected_module"),
        [
            (Task.EMBEDDING, "ai_workers.workers.embedding"),
            (Task.RERANKER_LLM, "ai_workers.workers.reranker"),
            (Task.VL_EMBEDDING, "ai_workers.workers.vl_embedding"),
            (Task.VL_RERANKER, "ai_workers.workers.vl_reranker"),
            (Task.OCR, "ai_workers.workers.ocr"),
            (Task.TTS, "ai_workers.workers.tts"),
            (Task.ASR, "ai_workers.workers.asr"),
        ],
    )
    def test_task_module_mapping(self, task: Task, expected_module: str) -> None:
        models = list_models(task=task)
        for model in models:
            assert model.worker_module == expected_module

    @pytest.mark.parametrize(
        ("model_name", "expected_var"),
        [
            ("qwen3-embedding-0.6b", "embedding_app"),
            ("qwen3-embedding-8b", "embedding_app"),
            ("qwen3-reranker-8b", "reranker_app"),
            ("qwen3-vl-embedding-2b", "vl_embedding_app"),
            ("qwen3-vl-embedding-8b", "vl_embedding_app"),
            ("qwen3-vl-reranker-8b", "vl_reranker_app"),
            ("deepseek-ocr-2", "ocr_app"),
            ("qwen3-tts-0.6b", "tts_app"),
            ("qwen3-tts-1.7b", "tts_app"),
            ("qwen3-asr-0.6b", "asr_app"),
            ("qwen3-asr-1.7b", "asr_app"),
        ],
    )
    def test_modal_app_var_mapping(self, model_name: str, expected_var: str) -> None:
        """Each model must point to the correct app variable in its worker module."""
        config = get_model(model_name)
        assert config.modal_app_var == expected_var


def test_model_registry_mm_reranker_entry():
    config = MODEL_REGISTRY["gemma4-reranker-v1"]
    assert config.task == Task.MM_RERANKER
    assert config.worker_module == "ai_workers.workers.mm_reranker"
    assert config.modal_app_var == "mm_reranker_app"
