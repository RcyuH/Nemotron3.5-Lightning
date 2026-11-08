#!/usr/bin/env python3
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

"""Heal Lightning's MTP block from local HF or native Megatron weights."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from megatron.bridge.training.callbacks import CallbackContext
    from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training inputs; keep help and schema preflight usable without CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--hf-model-path",
        type=Path,
        help="Matching local HF config/tokenizer snapshot; required for Megatron weights.",
    )
    parser.add_argument("--train-data", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-format", choices=("messages", "input-output"), default="messages")
    parser.add_argument("--loss-mode", choices=("last_turn", "assistant"), default="last_turn")
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--train-iters", type=int, default=1000)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep", type=int, default=16)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume this output directory's native checkpoint.")
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Check paths, HF metadata and every JSONL row without importing the training stack.",
    )
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> int:
    """Validate local input schemas; return row count (zero outside preflight).

    This is deliberately not a tokenizer, weight-integrity or GPU smoke test.
    The native checkpoint loader validates required weights during training.
    """
    for key in ("seq_length", "train_iters", "global_batch_size", "micro_batch_size", "ep", "cp", "save_interval"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not 0 <= args.warmup_iters < args.train_iters:
        raise ValueError("Require 0 <= warmup-iters < train-iters")
    if 128 % args.ep:
        raise ValueError("ep must divide Lightning's 128 experts")
    if args.seq_length % (128 * args.cp):
        raise ValueError("seq-length must be divisible by 128 * cp for this padded Mamba recipe")
    if not args.pretrained_checkpoint.is_dir():
        raise ValueError("pretrained-checkpoint must be an existing local directory")
    if args.hf_model_path is None:
        args.hf_model_path = args.pretrained_checkpoint
    config_path = args.hf_model_path / "config.json"
    if not config_path.is_file() or not (args.hf_model_path / "tokenizer_config.json").is_file():
        raise ValueError("hf-model-path must contain the matching HF config.json and tokenizer_config.json")
    hf_config = json.loads(config_path.read_text())
    if hf_config.get("model_type") != "nemotron_h" or (
        hf_config.get("num_hidden_layers"),
        hf_config.get("hidden_size"),
        hf_config.get("n_routed_experts"),
    ) != (52, 2688, 128):
        raise ValueError("HF metadata must describe Nemotron 3.5 Lightning 30B-A3B")
    if not hf_config.get("num_nextn_predict_layers"):
        raise ValueError("HF metadata has no MTP head; restore compatible MTP config AND weights before healing")
    if args.pretrained_checkpoint.resolve() == args.output_dir.resolve():
        raise ValueError("Use a separate output-dir, not the source checkpoint directory")
    checkpoint_dir = args.output_dir / "checkpoints"
    if args.resume:
        if not (checkpoint_dir / "latest_checkpointed_iteration.txt").is_file():
            raise ValueError("--resume requires output-dir/checkpoints/latest_checkpointed_iteration.txt")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is not empty; use a new directory or --resume for an existing healing run")
    for path in args.train_data:
        if not path.is_file():
            raise ValueError(f"Missing training JSONL: {path}")
    # torchrun supplies WORLD_SIZE. PP=1 and ETP=1 are fixed in this entry point.
    if "WORLD_SIZE" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        if world <= 0 or world % (args.tp * args.cp) or world % args.ep:
            raise ValueError("WORLD_SIZE must be divisible by both tp*cp and ep")
        dp = world // (args.tp * args.cp)
        if args.global_batch_size % (dp * args.micro_batch_size):
            raise ValueError("global-batch-size must be divisible by DP * micro-batch-size")
    if not args.check_inputs:
        return 0
    count = 0
    for path in args.train_data:
        file_count = 0
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: each row must be an object")
                if args.data_format == "messages":
                    messages = row.get("messages")
                    if (
                        not isinstance(messages, list)
                        or not messages
                        or any(not isinstance(message, dict) or "role" not in message for message in messages)
                    ):
                        raise ValueError(f"{path}:{line_number}: expected a nonempty messages list")
                    if messages[-1]["role"] != "assistant":
                        raise ValueError(
                            f"{path}:{line_number}: rollout row must end with the generated assistant turn"
                        )
                    answer = messages[-1]
                    if not any(answer.get(key) for key in ("content", "reasoning_content", "tool_calls")):
                        raise ValueError(f"{path}:{line_number}: empty assistant response")
                elif not all(isinstance(row.get(key), str) and row[key] for key in ("input", "output")):
                    raise ValueError(f"{path}:{line_number}: expected nonempty input/output strings")
                count += 1
                file_count += 1
        if not file_count:
            raise ValueError(f"Training JSONL is empty: {path}")
    return count


def build_config(args: argparse.Namespace) -> ConfigContainer:
    """Apply explicit run inputs to the MTP-only recipe."""
    from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, HFDatasetSourceConfig
    from megatron.bridge.data.sft_processing import PromptCompletionSFTPreprocessingConfig
    from megatron.bridge.recipes.nemotronh import nemotron_3_5_lightning_mtp_healing_config

    cfg = nemotron_3_5_lightning_mtp_healing_config(hf_model_path=str(args.hf_model_path))
    cfg.model.seq_length = args.seq_length
    cfg.model.tensor_model_parallel_size = args.tp
    cfg.model.sequence_parallel = args.tp > 1
    cfg.model.expert_model_parallel_size = args.ep
    cfg.model.context_parallel_size = args.cp
    cfg.model.cp_comm_type = "p2p"
    cfg.dataset.seq_length = args.seq_length
    cfg.dataset.source = HFDatasetSourceConfig(
        path_or_dataset="json", split="train", load_kwargs={"data_files": [str(p) for p in args.train_data]}
    )
    if args.data_format == "messages":
        cfg.dataset.preprocessing = ChatSFTPreprocessingConfig(loss_mode=args.loss_mode)
    else:
        # Input is the exact rendered generation prefix, output the sampled
        # continuation. Never insert the old recipe's extra separating space.
        cfg.dataset.preprocessing = PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator="",
            strip_whitespace=False,
            add_eos=False,
            loss_mode="completion",
        )
    cfg.train.train_iters = args.train_iters
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.optimizer.lr = args.lr
    cfg.scheduler.lr_warmup_iters = args.warmup_iters
    cfg.scheduler.lr_decay_iters = args.train_iters
    cfg.checkpoint.pretrained_checkpoint = str(args.pretrained_checkpoint)
    cfg.checkpoint.save = str(args.output_dir / "checkpoints")
    cfg.checkpoint.load = cfg.checkpoint.save if args.resume else None
    # pretrained_checkpoint starts at step zero with new optimizer/RNG state;
    # checkpoint.load resumes the complete healing state. Do not force finetune
    # or no_load_optim here: those switches would break resume semantics.
    cfg.checkpoint.save_interval = args.save_interval
    cfg.scheduler.override_opt_param_scheduler = not args.resume
    cfg.logger.tensorboard_dir = str(args.output_dir / "tb_logs")
    cfg.logger.log_interval = 1
    return cfg


def verify_mtp_only(context: CallbackContext) -> None:
    """Fail before step one if the actual model did not freeze as requested."""
    trainable = 0
    frozen = 0
    for chunk in context.model:
        for name, parameter in chunk.named_parameters():
            is_mtp = "mtp.layers." in name
            if parameter.requires_grad != is_mtp:
                raise RuntimeError(f"Unexpected requires_grad={parameter.requires_grad}: {name}")
            if parameter.requires_grad:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        for name, module in chunk.named_modules():
            if hasattr(module, "expert_bias") and "mtp.layers." not in name:
                if not getattr(module, "frozen_expert_bias", False):
                    raise RuntimeError(f"Backbone router expert_bias is not frozen: {name}")
    if not trainable or not frozen:
        raise RuntimeError("Expected both trainable MTP parameters and frozen backbone parameters on each PP=1 rank")
    logger.info("MTP freeze verified on this rank: %s trainable, %s frozen parameter elements", trainable, frozen)


def main() -> None:
    """Run schema preflight or distributed MTP healing."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    count = validate_inputs(args)
    if args.check_inputs:
        logger.info(
            "Input preflight passed: %d rows. Weight loading, token masks and GPU training are not checked.", count
        )
        return
    if "WORLD_SIZE" not in os.environ:
        raise ValueError(
            "Launch training with uv run python -m torch.distributed.run; use --check-inputs for CPU preflight"
        )
    from megatron.bridge.training.callbacks import CallbackManager
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    cfg = build_config(args)
    logger.info("MTP-only healing: EP=%d TP=%d CP=%d; output=%s", args.ep, args.tp, args.cp, args.output_dir)
    callbacks = CallbackManager()
    callbacks.register("on_train_start", verify_mtp_only)
    finetune(config=cfg, forward_step_func=forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()
