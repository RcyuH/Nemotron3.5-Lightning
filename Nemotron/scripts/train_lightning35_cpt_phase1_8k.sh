#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch Nemotron 3.5 Lightning CPT phase 1 through the Nemotron YAML wrapper.
# Megatron Bridge remains the training engine. This launcher intentionally uses
# torchrun and enforces offline Hugging Face / Transformers / W&B operation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMOTRON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-$(cd "${NEMOTRON_ROOT}/../Megatron-Bridge" && pwd)}"
CONFIG_PATH="${NEMOTRON_ROOT}/src/nemotron/recipes/lightning35/stage0_pretrain/config/cpt_phase1_8k.yaml"
TRAIN_SCRIPT="${NEMOTRON_ROOT}/src/nemotron/recipes/lightning35/stage0_pretrain/train.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${CPT_DATA_BLEND:?Set CPT_DATA_BLEND to the absolute per-split blend.json path}"
: "${CPT_DATA_CACHE:?Set CPT_DATA_CACHE to a shared dataset-cache directory}"
: "${CPT_HF_MODEL_DIR:?Set CPT_HF_MODEL_DIR to the local Base HF snapshot}"
: "${CPT_BASE_CHECKPOINT:?Set CPT_BASE_CHECKPOINT to a native checkpoint or local HF snapshot}"
: "${CPT_RUN_ROOT:?Set CPT_RUN_ROOT to a shared output directory}"

for required_path in "${CPT_DATA_BLEND}" "${CPT_HF_MODEL_DIR}" "${CPT_BASE_CHECKPOINT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

mkdir -p "${CPT_DATA_CACHE}" "${CPT_RUN_ROOT}"

export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}/src:${MEGATRON_BRIDGE_ROOT}/3rdparty/Megatron-LM:${NEMOTRON_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${CPT_RUN_ROOT}/hf-cache}"
export NEMO_HOME="${NEMO_HOME:-${CPT_RUN_ROOT}/nemo-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CPT_RUN_ROOT}/flashinfer-workspace}"
export FLASHINFER_CUBIN_DIR="${FLASHINFER_CUBIN_DIR:-${CPT_RUN_ROOT}/flashinfer-cubins}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HOME}" "${NEMO_HOME}" "${FLASHINFER_WORKSPACE_BASE}" "${FLASHINFER_CUBIN_DIR}"

TRAIN_ARGS=(
    "${TRAIN_SCRIPT}"
    --config "${CONFIG_PATH}"
    "$@"
)

echo "Config:          ${CONFIG_PATH}"
echo "Data blend:      ${CPT_DATA_BLEND}"
echo "Local HF model:  ${CPT_HF_MODEL_DIR}"
echo "Base checkpoint: ${CPT_BASE_CHECKPOINT}"
echo "Run root:        ${CPT_RUN_ROOT}"
echo "Token target:    200,001,191,936 (47,684 x 8,192 x 512)"

# Single-node defaults. For multi-node torchrun, launch this same script once
# per node with identical NNODES/MASTER_ADDR/MASTER_PORT and a unique NODE_RANK.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ARGS[@]}"
