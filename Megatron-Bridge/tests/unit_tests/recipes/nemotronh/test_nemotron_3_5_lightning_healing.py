"""Recipe contract tests for the CUDA/Megatron Bridge environment."""

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.data.builders import ChatSFTPreprocessingConfig
from megatron.bridge.recipes.nemotronh import nemotron_3_5_lightning_mtp_healing_config
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_5_lightning_pretrain_config


pytestmark = pytest.mark.unit


def test_healing_imports_custom_metadata_and_freezes_base():
    provider = nemotron_3_5_lightning_pretrain_config().model
    bridge = Mock()
    bridge.to_megatron_provider.return_value = provider
    with patch(
        "megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning.AutoBridge.from_hf_pretrained", return_value=bridge
    ) as load:
        cfg = nemotron_3_5_lightning_mtp_healing_config(hf_model_path="/checkpoints/post-trained-hf")
    load.assert_called_once_with("/checkpoints/post-trained-hf")
    bridge.to_megatron_provider.assert_called_once_with(load_weights=False)
    assert cfg.model.freeze_base_model_for_mtp
    assert cfg.model.mtp_detach_heads
    assert cfg.model.mtp_num_layers == 2
    assert cfg.model.mtp_use_repeated_layer
    assert cfg.model.mtp_hybrid_override_pattern == "*E"
    assert cfg.model.keep_mtp_spec_in_bf16
    assert cfg.peft is None
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.tokenizer.tokenizer_model == "/checkpoints/post-trained-hf"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {}
    assert isinstance(cfg.dataset.preprocessing, ChatSFTPreprocessingConfig)
    assert cfg.dataset.preprocessing.loss_mode == "last_turn"
    assert not cfg.dataset.enable_in_batch_packing
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.dist_ckpt_strictness == "raise_unexpected"


@pytest.mark.parametrize(
    "field,value", [("mtp_num_layers", 0), ("hidden_size", 4096), ("mtp_use_repeated_layer", False)]
)
def test_healing_rejects_incompatible_source(field, value):
    provider = nemotron_3_5_lightning_pretrain_config().model
    assert hasattr(provider, field)
    setattr(provider, field, value)
    bridge = Mock()
    bridge.to_megatron_provider.return_value = provider
    with patch(
        "megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning.AutoBridge.from_hf_pretrained", return_value=bridge
    ):
        with pytest.raises(ValueError):
            nemotron_3_5_lightning_mtp_healing_config(hf_model_path="/checkpoints/post-trained-hf")
