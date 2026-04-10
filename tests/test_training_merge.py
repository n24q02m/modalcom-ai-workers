import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import transformers
from training.gemma4_reranker.merge import MergeConfig, merge_and_push, verify_merged_model

# Inject stubs for heavy dependencies
peft_stub = types.ModuleType("peft")
peft_stub.PeftModel = MagicMock()
sys.modules["peft"] = peft_stub

# Gemma4ForConditionalGeneration is not in the transformers stub in conftest.py
transformers.Gemma4ForConditionalGeneration = MagicMock()


@pytest.fixture
def merge_config(tmp_path):
    return MergeConfig(
        adapter_path=tmp_path / "adapter",
        base_model_id="google/gemma-4-E4B-it",
        hub_repo_id="n24q02m/gemma4-e4b-reranker-v1",
        stage=1,
        output_dir=tmp_path / "merged",
        push=True,
    )


def test_merge_and_push(merge_config):
    # Mocking external components
    with (
        patch("torch.float16", "float16"),
        patch("torch.bfloat16", "bfloat16"),
        patch("peft.PeftModel.from_pretrained") as mock_peft_from_pretrained,
        patch("transformers.AutoProcessor.from_pretrained") as mock_processor_from_pretrained,
        patch(
            "transformers.Gemma4ForConditionalGeneration.from_pretrained"
        ) as mock_model_from_pretrained,
    ):
        # Setup mocks
        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model

        mock_peft_model = MagicMock()
        mock_peft_from_pretrained.return_value = mock_peft_model

        mock_merged_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_merged_model.to.return_value = mock_merged_model

        mock_processor = MagicMock()
        mock_processor_from_pretrained.return_value = mock_processor

        # Execute function
        result_path = merge_and_push(merge_config)

        # Assertions
        assert result_path == Path(merge_config.output_dir)
        mock_model_from_pretrained.assert_called_once_with(
            merge_config.base_model_id,
            torch_dtype="float16",
            device_map="cpu",
            trust_remote_code=True,
        )
        mock_peft_from_pretrained.assert_called_once_with(
            mock_model, str(merge_config.adapter_path)
        )
        mock_peft_model.merge_and_unload.assert_called_once()
        mock_merged_model.to.assert_called_once_with("bfloat16")
        mock_merged_model.save_pretrained.assert_called_once_with(
            str(merge_config.output_dir),
            safe_serialization=True,
        )
        mock_processor_from_pretrained.assert_called_once_with(
            merge_config.base_model_id,
            trust_remote_code=True,
        )
        mock_processor.save_pretrained.assert_called_once_with(str(merge_config.output_dir))

        # Verify push
        mock_merged_model.push_to_hub.assert_called_once_with(
            merge_config.hub_repo_id,
            commit_message=f"feat: stage {merge_config.stage} merged checkpoint",
            safe_serialization=True,
        )
        mock_processor.push_to_hub.assert_called_once_with(merge_config.hub_repo_id)


def test_merge_and_push_no_push(merge_config):
    # Update config to disable push
    config = MergeConfig(
        adapter_path=merge_config.adapter_path,
        base_model_id=merge_config.base_model_id,
        hub_repo_id=merge_config.hub_repo_id,
        stage=merge_config.stage,
        output_dir=merge_config.output_dir,
        push=False,
    )

    # Mocking external components
    with (
        patch("torch.float16", "float16"),
        patch("torch.bfloat16", "bfloat16"),
        patch("peft.PeftModel.from_pretrained") as mock_peft_from_pretrained,
        patch("transformers.AutoProcessor.from_pretrained") as mock_processor_from_pretrained,
        patch(
            "transformers.Gemma4ForConditionalGeneration.from_pretrained"
        ) as mock_model_from_pretrained,
    ):
        # Setup mocks
        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_peft_model = MagicMock()
        mock_peft_from_pretrained.return_value = mock_peft_model
        mock_merged_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_merged_model.to.return_value = mock_merged_model
        mock_processor = MagicMock()
        mock_processor_from_pretrained.return_value = mock_processor

        # Execute function
        merge_and_push(config)

        # Assert push was NOT called
        mock_merged_model.push_to_hub.assert_not_called()
        mock_processor.push_to_hub.assert_not_called()


def test_verify_merged_model(tmp_path):
    model_path = tmp_path / "merged"
    model_path.mkdir()

    with (
        patch("torch.bfloat16", "bfloat16"),
        patch("transformers.AutoProcessor.from_pretrained") as mock_processor_from_pretrained,
        patch(
            "transformers.AutoModelForImageTextToText.from_pretrained"
        ) as mock_model_from_pretrained,
    ):
        # Setup mocks
        mock_processor = MagicMock()
        mock_processor_from_pretrained.return_value = mock_processor
        mock_processor.tokenizer.convert_tokens_to_ids.side_effect = lambda x: (
            100 if x == "yes" else 101
        )
        mock_processor.return_value = {"input_ids": [1, 2, 3]}

        mock_model = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_model.parameters.return_value = iter([MagicMock(dtype="bfloat16")])

        mock_outputs = MagicMock()
        mock_outputs.logits.shape = (1, 10, 50000)
        mock_model.return_value = mock_outputs

        # Execute function
        results = verify_merged_model(model_path)

        # Assertions
        assert results["status"] == "ok"
        assert results["dtype"] == "bfloat16"
        assert results["vocab_size"] == 50000
        assert results["yes_token_id"] == 100
        assert results["no_token_id"] == 101
        assert results["logits_shape"] == [1, 10, 50000]

        mock_processor_from_pretrained.assert_called_once_with(
            str(model_path), trust_remote_code=True
        )
        mock_model_from_pretrained.assert_called_once_with(
            str(model_path),
            torch_dtype="bfloat16",
            device_map="cpu",
            trust_remote_code=True,
        )
