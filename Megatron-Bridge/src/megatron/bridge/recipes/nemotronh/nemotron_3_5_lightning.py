# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MTP-only healing for post-trained Nemotron 3.5 Lightning checkpoints."""

from typing import cast

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, DirectHFSFTDatasetConfig, HFDatasetSourceConfig
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer


NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def nemotron_3_5_lightning_mtp_healing_config(
    *, hf_model_path: str = NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID
) -> ConfigContainer:
    """Build a BF16 MTP-healing recipe, initially mapped to 16 GPUs (EP=16).

    Load architecture and tokenizer metadata from the actual post-trained HF
    snapshot. Weights are loaded later through ``checkpoint.pretrained_checkpoint``.
    Only ``mtp.layers.*`` is optimized; the shared embedding, LM head, backbone,
    and backbone router expert biases stay frozen by Bridge's native setup hook.

    Args:
        hf_model_path: HF snapshot containing config/tokenizer files, including
            the original Lightning MTP configuration. Also used with a native
            Megatron weight checkpoint. Prefer a local snapshot for offline jobs.

    Returns:
        Config requiring a checkpoint and a rollout data source before training.
        LR, schedule and MTP loss weight are starting settings, not paper values.

    Raises:
        ValueError: The source is not a compatible Lightning model with MTP.
    """
    cfg = _sft_common()
    model = AutoBridge.from_hf_pretrained(hf_model_path).to_megatron_provider(load_weights=False)
    if not isinstance(model, HybridModelProvider):
        raise ValueError("MTP healing requires a Nemotron-H HybridModelProvider.")
    if (model.num_layers, model.hidden_size, model.num_moe_experts) != (52, 2688, 128):
        raise ValueError("Expected the Nemotron 3.5 Lightning 30B-A3B architecture.")
    if not model.mtp_num_layers or model.mtp_hybrid_override_pattern != "*E":
        raise ValueError("Checkpoint config must retain Lightning's attention+MoE MTP block (*E).")
    if not model.mtp_use_repeated_layer:
        raise ValueError("Expected Lightning's shared/repeated MTP block.")
    cfg.model = cast(HybridModelProvider, model)

    # Same repeated-head objective as the Lightning SFT recipe and old Super code.
    model.mtp_num_layers = 2
    model.mtp_use_repeated_layer = True
    model.keep_mtp_spec_in_bf16 = True
    model.mtp_loss_scaling_factor = 0.3
    model.mtp_detach_heads = True
    model.freeze_base_model_for_mtp = True
    model.calculate_per_token_loss = True

    # A conservative 2 x 8 B200 layout. All-to-all does not require DeepEP.
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.expert_model_parallel_size = 16
    model.expert_tensor_parallel_size = 1
    model.sequence_parallel = False
    model.seq_length = 4096
    model.moe_token_dispatcher_type = "alltoall"
    model.moe_shared_expert_overlap = False
    model.moe_router_force_load_balancing = False
    model.apply_rope_fusion = False
    model.attention_backend = "fused"
    model.use_fused_weighted_squared_relu = True
    model.cuda_graph_impl = "none"
    model.recompute_granularity = None
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    # A complete response per row by default; no cross-conversation packing.
    cfg.dataset = DirectHFSFTDatasetConfig(
        seq_length=model.seq_length,
        source=HFDatasetSourceConfig(path_or_dataset="json", split="train"),
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="last_turn"),
        do_validation=False,
        do_test=False,
        pad_to_max_length=True,
        enable_in_batch_packing=False,
    )
    cfg.validation.eval_iters = 0
    cfg.tokenizer.tokenizer_model = hf_model_path
    cfg.tokenizer.hf_tokenizer_kwargs = {}
    cfg.checkpoint.hf_source_path = hf_model_path
    # Reject tensors required by this model but absent from the checkpoint;
    # allow unused source optimizer tensors when initializing from a full-SFT run.
    cfg.checkpoint.dist_ckpt_strictness = "raise_unexpected"
    cfg.checkpoint.load = None
    cfg.peft = None
    return cfg


__all__ = ["nemotron_3_5_lightning_mtp_healing_config"]
