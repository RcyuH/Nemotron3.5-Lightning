#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline MTP healing for post-trained Nemotron 3.5 Lightning, 2 x 8 B200.

Standalone entry point: no dependency on the custom recipe from this chat.
Requires the current Bridge/Core APIs: AutoBridge, DirectHFSFTDatasetConfig,
freeze_base_model_for_mtp, mtp_detach_heads and native backbone-router freezing.

Defaults follow the supplied Super training script: 8K, 7 repeated MTP depths,
MTP loss coefficient 1, GBS 64, 2 epoch-equivalents, LR 1e-5, no weight decay,
no packing, dynamic padding, last assistant turn only, no distributed optimizer.
The epoch-equivalent budget is floor(dataset_len / GBS) * epochs, as in the sample;
it is not an exact-epoch sampler contract. --dataset-len must be the actual row count.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if TYPE_CHECKING:
    from megatron.bridge.training.callbacks import CallbackContext
    from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse local paths and controls using the same CLI names as the sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-rl-checkpoint",
        required=True,
        type=Path,
        help="Local HF or native Megatron weights.",
    )
    parser.add_argument(
        "--local-hf",
        required=True,
        type=Path,
        help="Matching local HF config and tokenizer snapshot.",
    )
    parser.add_argument("--run-checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--tensorboard_dir",
        "--tensorboard-dir",
        dest="tensorboard_dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="training.jsonl and optional validation.jsonl.",
    )
    parser.add_argument("--dataset-len", required=True, type=int)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--train-iters",
        type=int,
        help="Optional explicit step budget, e.g. 5 for a smoke test.",
    )
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--no-validation", dest="do_validation", action="store_false", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-format", choices=("messages", "input-output"), default="messages")
    parser.add_argument("--loss-mode", choices=("last_turn", "assistant"), default="last_turn")
    parser.add_argument(
        "--mtp-num-layers",
        type=int,
        default=7,
        help="Repeated prediction depths, not independent blocks.",
    )
    parser.add_argument("--mtp-loss-scaling-factor", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-iters", type=int, default=30)
    parser.add_argument("--ep", type=int, default=16)
    parser.add_argument(
        "--distributed-optimizer",
        action="store_true",
        help="Shard optimizer state; off as in the sample.",
    )
    parser.add_argument(
        "--skip-nonfinite",
        action="store_true",
        help="Requires the local Bridge patch implementing train.skip_nonfinite_loss_and_grad.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[int, int]:
    """Validate inputs and return (iterations_per_epoch, total_train_iters)."""
    for field in (
        "dataset_len",
        "global_batch_size",
        "epochs",
        "seq_length",
        "mtp_num_layers",
        "ep",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.seq_length % 128:
        raise ValueError("seq-length must be a multiple of 128 for the Mamba padding configuration")
    if 128 % args.ep:
        raise ValueError("ep must divide Lightning's 128 experts")
    for field in ("lr", "mtp_loss_scaling_factor"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be positive and finite")
    if not math.isfinite(args.min_lr) or not 0 <= args.min_lr <= args.lr:
        raise ValueError("Require 0 <= min-lr <= lr")
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be nonnegative")
    if args.do_validation and args.eval_iters <= 0:
        raise ValueError("eval-iters must be positive when validation is enabled")
    iterations_per_epoch = args.dataset_len // args.global_batch_size
    if not iterations_per_epoch:
        raise ValueError("dataset-len must be at least global-batch-size")
    train_iters = args.train_iters if args.train_iters is not None else iterations_per_epoch * args.epochs
    if train_iters <= 0 or args.warmup_iters >= train_iters:
        raise ValueError("Require train-iters > warmup-iters; for a 5-step smoke test use --warmup-iters 1")
    # With TP=PP=CP=ETP=1, dense DP equals WORLD_SIZE; MBS is fixed at 1.
    world_size = int(os.environ.get("WORLD_SIZE", "16"))
    if world_size <= 0 or world_size % args.ep or args.global_batch_size % world_size:
        raise ValueError("WORLD_SIZE must be divisible by ep, and global-batch-size must be divisible by WORLD_SIZE")
    for filename in ("config.json", "tokenizer_config.json"):
        if not (args.local_hf / filename).is_file():
            raise FileNotFoundError(args.local_hf / filename)
    hf_config = json.loads((args.local_hf / "config.json").read_text())
    if hf_config.get("model_type") != "nemotron_h" or (
        hf_config.get("num_hidden_layers"),
        hf_config.get("hidden_size"),
        hf_config.get("n_routed_experts"),
    ) != (52, 2688, 128):
        raise ValueError("local-hf must describe Nemotron 3.5 Lightning 30B-A3B")
    if not hf_config.get("num_nextn_predict_layers"):
        raise ValueError("HF config has no MTP head; restore compatible MTP metadata AND weights before healing")
    for filename in ["training.jsonl"] + (["validation.jsonl"] if args.do_validation else []):
        if not (args.dataset_root / filename).is_file():
            raise FileNotFoundError(args.dataset_root / filename)
    checkpoint = args.run_checkpoint_dir if args.resume else args.post_rl_checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if args.run_checkpoint_dir.resolve() == args.post_rl_checkpoint.resolve():
        raise ValueError("run-checkpoint-dir must differ from post-rl-checkpoint")
    if args.resume:
        if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
            raise ValueError(
                "Resume requires a native Megatron checkpoint root with latest_checkpointed_iteration.txt"
            )
    elif args.run_checkpoint_dir.exists() and any(args.run_checkpoint_dir.iterdir()):
        raise ValueError("Use an empty run-checkpoint-dir for a new run, or --resume for an existing healing run")
    return iterations_per_epoch, train_iters


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Build a local, BF16, MTP-only configuration without modifying recipe globals."""
    iterations_per_epoch, train_iters = validate_args(args)
    from megatron.bridge import AutoBridge
    from megatron.bridge.data.builders import (
        ChatSFTPreprocessingConfig,
        DirectHFSFTDatasetConfig,
        HFDatasetSourceConfig,
    )
    from megatron.bridge.data.sft_processing import (
        PromptCompletionSFTPreprocessingConfig,
    )
    from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
    from megatron.bridge.recipes.common import _sft_common

    cfg = _sft_common()
    model = AutoBridge.from_hf_pretrained(str(args.local_hf)).to_megatron_provider(load_weights=False)
    if not isinstance(model, HybridModelProvider) or model.mtp_hybrid_override_pattern != "*E":
        raise ValueError("Expected Lightning's hybrid provider with an attention+MoE (*E) MTP block")
    if not model.mtp_num_layers or not model.mtp_use_repeated_layer:
        raise ValueError("Checkpoint must retain the shared/repeated Lightning MTP block")
    if not hasattr(model, "freeze_base_model_for_mtp") or not hasattr(model, "mtp_detach_heads"):
        raise RuntimeError("This script needs Bridge/Core with native freeze_base_model_for_mtp and mtp_detach_heads")
    cfg.model = model
    model.seq_length = args.seq_length
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.expert_model_parallel_size = args.ep
    model.expert_tensor_parallel_size = 1
    model.sequence_parallel = False
    model.calculate_per_token_loss = True
    model.mtp_num_layers = args.mtp_num_layers
    model.mtp_use_repeated_layer = True
    model.keep_mtp_spec_in_bf16 = True
    model.mtp_loss_scaling_factor = args.mtp_loss_scaling_factor
    model.mtp_detach_heads = True
    # Bridge registers its freeze hook before DDP/optimizer construction. It
    # freezes all non-MTP parameters AND backbone router expert-bias updates.
    model.freeze_base_model_for_mtp = True
    model.moe_router_load_balancing_type = "none"
    model.moe_aux_loss_coeff = 0.0
    model.moe_router_force_load_balancing = False
    model.moe_token_dispatcher_type = "alltoall"
    model.moe_shared_expert_overlap = False
    model.apply_rope_fusion = False
    model.attention_backend = "fused"
    model.use_fused_weighted_squared_relu = True
    model.cuda_graph_impl = "none"
    model.recompute_granularity = None
    model.recompute_modules = None
    cfg.tokenizer.tokenizer_model = str(args.local_hf)
    cfg.tokenizer.hf_tokenizer_kwargs = {}
    cfg.checkpoint.hf_source_path = str(args.local_hf)
    cfg.peft = None
    cfg.rng.seed = 1234

    def jsonl_source(filename: str) -> HFDatasetSourceConfig:
        return HFDatasetSourceConfig(
            path_or_dataset="json",
            split="train",
            load_kwargs={"data_files": str(args.dataset_root / filename)},
        )

    cfg.dataset = DirectHFSFTDatasetConfig(
        source=jsonl_source("training.jsonl"),
        validation_source=jsonl_source("validation.jsonl") if args.do_validation else None,
        seq_length=args.seq_length,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode=args.loss_mode),
        do_validation=args.do_validation,
        do_test=False,
        enable_in_batch_packing=False,
        pad_to_max_length=False,
        pad_to_multiple_of=128,
        dataloader_type="cyclic",
        num_workers=2,
    )
    if args.data_format == "input-output":
        # Input must be the fully rendered generation prefix; output the exact
        # generated continuation, including the desired ending token. No extra
        # space, whitespace stripping, template application or EOS insertion.
        cfg.dataset.preprocessing = PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator="",
            strip_whitespace=False,
            add_eos=False,
            loss_mode="completion",
        )
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = iterations_per_epoch if args.do_validation else None
    cfg.validation.eval_iters = args.eval_iters if args.do_validation else 0
    cfg.checkpoint.save_interval = min(iterations_per_epoch, train_iters)

    cfg.checkpoint.load = str(args.run_checkpoint_dir) if args.resume else None
    cfg.checkpoint.pretrained_checkpoint = None if args.resume else str(args.post_rl_checkpoint)
    cfg.checkpoint.finetune = not args.resume
    cfg.checkpoint.load_optim = args.resume
    cfg.checkpoint.load_rng = args.resume
    cfg.checkpoint.save = str(args.run_checkpoint_dir)
    cfg.checkpoint.dist_ckpt_strictness = "raise_unexpected"
    cfg.checkpoint.async_save = False
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.ckpt_assume_constant_structure = False

    cfg.optimizer.lr = args.lr
    cfg.optimizer.min_lr = args.min_lr
    cfg.optimizer.decoupled_weight_decay = True
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.0
    cfg.scheduler.start_weight_decay = 0.0
    cfg.scheduler.end_weight_decay = 0.0
    cfg.scheduler.lr_warmup_iters = args.warmup_iters
    cfg.scheduler.lr_decay_iters = train_iters
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.override_opt_param_scheduler = not args.resume
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = args.distributed_optimizer
    cfg.optimizer.use_distributed_optimizer = args.distributed_optimizer
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    if args.skip_nonfinite:
        if not hasattr(cfg.train, "skip_nonfinite_loss_and_grad"):
            raise RuntimeError("--skip-nonfinite requires the Bridge train.skip_nonfinite_loss_and_grad patch")
        cfg.train.skip_nonfinite_loss_and_grad = True

    cfg.logger.log_progress = True
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_log_interval = 1
    cfg.logger.tensorboard_dir = str(args.tensorboard_dir)
    cfg.logger.log_runtime_to_tensorboard = True
    cfg.dist.distributed_timeout_minutes = 30
    return cfg


def verify_mtp_only(context: CallbackContext) -> None:
    """Check the real trainability and router-freeze flags before the first step."""
    trainable = frozen = 0
    for chunk in context.model:
        for name, parameter in chunk.named_parameters():
            if parameter.requires_grad != ("mtp.layers." in name):
                raise RuntimeError(f"Incorrect MTP-only trainability: {name}")
            if parameter.requires_grad:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        for name, module in chunk.named_modules():
            if hasattr(module, "expert_bias") and "mtp.layers." not in name:
                if not getattr(module, "frozen_expert_bias", False):
                    raise RuntimeError(f"Backbone router is not frozen: {name}")
    if not trainable or not frozen:
        raise RuntimeError("PP=1 requires both trainable MTP and frozen backbone parameters on every rank")
    logger.info(
        "MTP freeze verified on rank %s: %s trainable, %s frozen elements",
        os.getenv("RANK", "0"),
        trainable,
        frozen,
    )


def main() -> None:
    """Launch healing in the environment already configured by the user."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch training.py with torchrun")
    cfg = build_config(args)
    from megatron.bridge.training.callbacks import CallbackManager
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.utils.common_utils import print_rank_0

    callbacks = CallbackManager()
    callbacks.register("on_train_start", verify_mtp_only)
    print_rank_0(
        f"Lightning MTP healing: steps={cfg.train.train_iters}, seq={args.seq_length}, "
        f"MTP depths={args.mtp_num_layers}, EP={args.ep}, save={args.run_checkpoint_dir}"
    )
    finetune(config=cfg, forward_step_func=forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()
