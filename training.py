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

"""MTP healing for Nemotron 3.5 Lightning."""

import argparse
import logging
import os
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    DirectHFSFTDatasetConfig,
    HFDatasetSourceConfig,
)
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)
_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def parse_args() -> argparse.Namespace:
    """Read training paths and the basic recipe overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-rl-checkpoint", required=True)
    parser.add_argument("--run-checkpoint-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-len", required=True, type=int)
    parser.add_argument("--local-hf", required=True, help="HF config and tokenizer for this Lightning checkpoint")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--no-validation", action="store_false", dest="do_validation", default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Build the Lightning MTP healing configuration."""
    iterations_per_epoch = args.dataset_len // args.global_batch_size
    train_iters = iterations_per_epoch * args.epochs

    cfg = _sft_common()
    model = AutoBridge.from_hf_pretrained(args.local_hf).to_megatron_provider(load_weights=False)
    cfg.model = model
    model.seq_length = args.seq_length
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.context_parallel_size = 1
    model.expert_model_parallel_size = 16
    model.expert_tensor_parallel_size = 1
    model.sequence_parallel = False
    model.calculate_per_token_loss = True
    model.mtp_num_layers = 7
    model.mtp_use_repeated_layer = True
    model.mtp_hybrid_override_pattern = "*E"
    model.keep_mtp_spec_in_bf16 = True
    model.mtp_loss_scaling_factor = 1.0
    model.mtp_detach_heads = True
    model.freeze_base_model_for_mtp = True
    model.moe_router_load_balancing_type = "none"
    model.moe_aux_loss_coeff = 0.0
    model.moe_token_dispatcher_type = "alltoall"
    model.moe_shared_expert_overlap = False
    model.cuda_graph_impl = "none"
    model.recompute_granularity = None
    model.recompute_modules = None

    cfg.dataset = DirectHFSFTDatasetConfig(
        source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="train",
            load_kwargs={"data_files": str(Path(args.dataset_root) / "training.jsonl")},
        ),
        validation_source=(
            HFDatasetSourceConfig(
                path_or_dataset="json",
                split="train",
                load_kwargs={"data_files": str(Path(args.dataset_root) / "validation.jsonl")},
            )
            if args.do_validation
            else None
        ),
        seq_length=args.seq_length,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="last_turn"),
        do_validation=args.do_validation,
        do_test=False,
        enable_in_batch_packing=False,
        pad_to_max_length=False,
        dataloader_type="cyclic",
        num_workers=2,
    )

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = iterations_per_epoch if args.do_validation else None
    cfg.validation.eval_iters = args.eval_iters if args.do_validation else 0

    cfg.tokenizer.tokenizer_model = args.local_hf
    cfg.tokenizer.hf_tokenizer_kwargs = {}
    cfg.checkpoint.hf_source_path = args.local_hf
    cfg.checkpoint.save = args.run_checkpoint_dir
    cfg.checkpoint.load = args.run_checkpoint_dir if args.resume else None
    cfg.checkpoint.pretrained_checkpoint = None if args.resume else args.post_rl_checkpoint
    cfg.checkpoint.finetune = not args.resume
    cfg.checkpoint.load_optim = args.resume
    cfg.checkpoint.load_rng = args.resume
    cfg.checkpoint.save_interval = iterations_per_epoch
    cfg.checkpoint.dist_ckpt_strictness = "raise_unexpected"
    cfg.checkpoint.async_save = False
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.ckpt_assume_constant_structure = False

    cfg.optimizer.lr = 1e-5
    cfg.optimizer.min_lr = 1e-6
    cfg.optimizer.decoupled_weight_decay = True
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.weight_decay = 0.0
    cfg.scheduler.start_weight_decay = 0.0
    cfg.scheduler.end_weight_decay = 0.0
    cfg.scheduler.lr_warmup_iters = 30
    cfg.scheduler.lr_decay_iters = train_iters
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.override_opt_param_scheduler = not args.resume

    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = False
    cfg.optimizer.use_distributed_optimizer = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    cfg.logger.log_progress = True
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_log_interval = 1
    cfg.logger.tensorboard_dir = args.tensorboard_dir
    cfg.dist.distributed_timeout_minutes = 30
    cfg.rng.seed = 1234
    cfg.train.skip_nonfinite_loss_and_grad = True
    return cfg


def main() -> None:
    """Start MTP healing."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = build_config(args)
    print_rank_0(
        f"Starting Lightning MTP healing: train_iters={cfg.train.train_iters}, "
        f"seq_length={cfg.model.seq_length}, checkpoint={cfg.checkpoint.save}"
    )
    finetune(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
