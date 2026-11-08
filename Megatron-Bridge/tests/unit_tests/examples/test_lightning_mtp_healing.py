"""CPU-only input and freeze-guard tests; no Megatron/CUDA imports required."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "examples/models/nemotron/nemotron_3/lightning/mtp_healing.py"
spec = importlib.util.spec_from_file_location("lightning_mtp_healing", SCRIPT)
healing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(healing)


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "num_hidden_layers": 52,
                "hidden_size": 2688,
                "n_routed_experts": 128,
                "num_nextn_predict_layers": 1,
            }
        )
    )
    (hf / "tokenizer_config.json").write_text("{}")
    data = tmp_path / "rollouts.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        )
        + "\n"
    )
    return healing.parse_args(
        [
            "--pretrained-checkpoint",
            str(hf),
            "--train-data",
            str(data),
            "--output-dir",
            str(tmp_path / "out"),
            "--check-inputs",
        ]
    )


def test_chat_preflight_selects_custom_hf(args):
    assert healing.validate_inputs(args) == 1
    assert args.hf_model_path == args.pretrained_checkpoint


def test_megatron_requires_hf_metadata(args, tmp_path):
    args.pretrained_checkpoint = tmp_path / "native"
    args.pretrained_checkpoint.mkdir()
    with pytest.raises(ValueError, match="matching HF"):
        healing.validate_inputs(args)


def test_megatron_accepts_separate_hf_metadata(args, tmp_path):
    args.hf_model_path = args.pretrained_checkpoint
    args.pretrained_checkpoint = tmp_path / "native"
    args.pretrained_checkpoint.mkdir()
    assert healing.validate_inputs(args) == 1


def test_rendered_input_output_preflight(args):
    args.data_format = "input-output"
    args.train_data[0].write_text(json.dumps({"input": "<prefix>\n", "output": " A<eos>"}) + "\n")
    assert healing.validate_inputs(args) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"role": "assistant", "content": "", "reasoning_content": "reason"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "tool"}}]},
    ],
)
def test_reasoning_and_tool_call_responses_are_allowed(args, response):
    args.train_data[0].write_text(json.dumps({"messages": [{"role": "user", "content": "Q"}, response]}) + "\n")
    assert healing.validate_inputs(args) == 1


@pytest.mark.parametrize(
    "row",
    [
        [],
        {"messages": []},
        {"messages": [None]},
        {"messages": [{"role": "user", "content": "Q"}]},
        {"messages": [{"role": "assistant", "content": ""}]},
    ],
)
def test_bad_chat_rows_rejected(args, row):
    args.train_data[0].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        healing.validate_inputs(args)


def test_empty_file_rejected(args):
    args.train_data[0].write_text("")
    with pytest.raises(ValueError, match="empty"):
        healing.validate_inputs(args)


def test_stripped_mtp_metadata_rejected(args):
    path = args.pretrained_checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["num_nextn_predict_layers"] = 0
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="no MTP head"):
        healing.validate_inputs(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ep", 3),
        ("cp", 0),
        ("seq_length", 123),
        ("lr", float("nan")),
        ("warmup_iters", 1000),
        ("micro_batch_size", 0),
    ],
)
def test_invalid_training_controls_rejected(args, field, value):
    assert hasattr(args, field)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        healing.validate_inputs(args)


def test_two_node_batch_layout(args, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    assert healing.validate_inputs(args) == 1
    args.global_batch_size = 17
    with pytest.raises(ValueError, match="DP"):
        healing.validate_inputs(args)


def test_invalid_expert_mesh_rejected(args, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        healing.validate_inputs(args)


def test_resume_must_be_explicit_and_native(args):
    checkpoint = args.output_dir / "checkpoints"
    checkpoint.mkdir(parents=True)
    with pytest.raises(ValueError, match="not empty"):
        healing.validate_inputs(args)
    args.resume = True
    with pytest.raises(ValueError, match="latest_checkpointed_iteration"):
        healing.validate_inputs(args)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("10")
    assert healing.validate_inputs(args) == 1


def _context(*, base_grad=False, mtp_grad=True, router_frozen=True):
    base = Mock(requires_grad=base_grad)
    base.numel.return_value = 10
    mtp = Mock(requires_grad=mtp_grad)
    mtp.numel.return_value = 2
    chunk = Mock()
    chunk.named_parameters.return_value = [
        ("module.decoder.weight", base),
        ("module.mtp.layers.0.weight", mtp),
    ]
    chunk.named_modules.return_value = [
        ("module.decoder.router", SimpleNamespace(expert_bias=object(), frozen_expert_bias=router_frozen))
    ]
    return SimpleNamespace(model=[chunk])


def test_runtime_freeze_guard_accepts_wrapped_names():
    healing.verify_mtp_only(_context())


@pytest.mark.parametrize("kwargs", [{"base_grad": True}, {"mtp_grad": False}, {"router_frozen": False}])
def test_runtime_freeze_guard_catches_backbone_or_router_updates(kwargs):
    with pytest.raises(RuntimeError):
        healing.verify_mtp_only(_context(**kwargs))
