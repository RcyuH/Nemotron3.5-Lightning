# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realign pre-packed SFT Parquet shards for context-parallel training.

The Lightning SFT collator removes one stored token from every logical
subsequence when it builds next-token inputs and labels. Consequently, a
stored subsequence of length ``L`` occupies ``L - 1`` runtime token positions.
For load-balanced context parallelism, every runtime subsequence must be
divisible by ``2 * CP``. Sequence parallelism may impose an additional
``CP * TP`` constraint.

This tool reads already-tokenized packed Parquet data, inserts zero-loss EOS
padding into each subsequence, and repacks the aligned subsequences without
running tokenization or a chat template again.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

PACKED_SCHEMA = pa.schema(
    [
        ("input_ids", pa.list_(pa.int32())),
        ("loss_mask", pa.list_(pa.uint8())),
        ("seq_start_id", pa.list_(pa.int32())),
    ]
)


@dataclass
class FileStats:
    """Counters produced while realigning one Parquet shard."""

    input_file: str
    output_file: str
    input_rows: int = 0
    output_rows: int = 0
    input_sequences: int = 0
    output_sequences: int = 0
    dropped_zero_runtime_sequences: int = 0
    original_runtime_tokens: int = 0
    padding_tokens_added: int = 0

    def merge(self, other: FileStats) -> None:
        """Add numeric counters from another shard."""
        self.input_rows += other.input_rows
        self.output_rows += other.output_rows
        self.input_sequences += other.input_sequences
        self.output_sequences += other.output_sequences
        self.dropped_zero_runtime_sequences += other.dropped_zero_runtime_sequences
        self.original_runtime_tokens += other.original_runtime_tokens
        self.padding_tokens_added += other.padding_tokens_added


def required_alignment(
    *,
    context_parallel_size: int,
    tensor_parallel_size: int,
    sequence_parallel: bool,
    explicit_multiple: int | None,
) -> int:
    """Return the packed runtime-length multiple required by the topology."""
    if context_parallel_size < 1:
        raise ValueError("context_parallel_size must be at least 1")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    if explicit_multiple is not None and explicit_multiple < 1:
        raise ValueError("pad_to_multiple must be at least 1")

    cp_multiple = 2 * context_parallel_size if context_parallel_size > 1 else 1
    sp_multiple = context_parallel_size * tensor_parallel_size if sequence_parallel and tensor_parallel_size > 1 else 1
    return math.lcm(explicit_multiple or 1, cp_multiple, sp_multiple)


def aligned_stored_length(stored_length: int, alignment: int) -> int:
    """Return stored length whose post-shift runtime length is aligned."""
    if stored_length < 1:
        raise ValueError(f"A stored subsequence must contain at least one token, got {stored_length}")
    runtime_length = stored_length - 1
    aligned_runtime_length = math.ceil(runtime_length / alignment) * alignment
    return aligned_runtime_length + 1


def iter_sequences(parquet_file: Path, *, input_batch_rows: int) -> Iterable[tuple[list[int], list[int]]]:
    """Yield logical subsequences from one packed Parquet shard."""
    source = pq.ParquetFile(parquet_file)
    missing = {"input_ids", "loss_mask", "seq_start_id"} - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"{parquet_file} is missing required columns: {sorted(missing)}")

    columns = ["input_ids", "loss_mask", "seq_start_id"]
    for batch in source.iter_batches(batch_size=input_batch_rows, columns=columns):
        input_rows = batch.column(0).to_pylist()
        mask_rows = batch.column(1).to_pylist()
        start_rows = batch.column(2).to_pylist()
        for row_index, (input_ids, loss_mask, starts) in enumerate(zip(input_rows, mask_rows, start_rows)):
            if len(input_ids) != len(loss_mask):
                raise ValueError(
                    f"{parquet_file}: batch row {row_index} has len(input_ids)={len(input_ids)} "
                    f"but len(loss_mask)={len(loss_mask)}"
                )
            if not starts or starts[0] != 0:
                raise ValueError(f"{parquet_file}: batch row {row_index} must have seq_start_id beginning at 0")

            boundaries = [*starts, len(input_ids)]
            for seq_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                if start < 0 or end <= start or end > len(input_ids):
                    raise ValueError(
                        f"{parquet_file}: invalid sequence boundary at batch row {row_index}, "
                        f"sequence {seq_index}: [{start}, {end})"
                    )
                yield input_ids[start:end], loss_mask[start:end]


class PackedShardWriter:
    """Incrementally write packed SFT rows with bounded host memory."""

    def __init__(
        self,
        output_path: Path,
        *,
        compression: str,
        output_row_group_size: int,
    ) -> None:
        self.output_path = output_path
        self.compression = compression
        self.output_row_group_size = output_row_group_size
        self._writer: pq.ParquetWriter | None = None
        self._input_ids: list[list[int]] = []
        self._loss_masks: list[list[int]] = []
        self._seq_starts: list[list[int]] = []

    def append(self, input_ids: list[int], loss_mask: list[int], seq_start_id: list[int]) -> None:
        """Append one packed row and flush complete output row groups."""
        self._input_ids.append(input_ids)
        self._loss_masks.append(loss_mask)
        self._seq_starts.append(seq_start_id)
        if len(self._input_ids) >= self.output_row_group_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows to disk."""
        if not self._input_ids:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(self._input_ids, type=PACKED_SCHEMA.field("input_ids").type),
                pa.array(self._loss_masks, type=PACKED_SCHEMA.field("loss_mask").type),
                pa.array(self._seq_starts, type=PACKED_SCHEMA.field("seq_start_id").type),
            ],
            schema=PACKED_SCHEMA,
        )
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.output_path,
                PACKED_SCHEMA,
                compression=None if self.compression == "none" else self.compression,
            )
        self._writer.write_table(table, row_group_size=self.output_row_group_size)
        self._input_ids.clear()
        self._loss_masks.clear()
        self._seq_starts.clear()

    def close(self) -> None:
        """Flush buffered rows and close the Parquet writer."""
        self.flush()
        if self._writer is None:
            pq.write_table(
                pa.Table.from_arrays(
                    [pa.array([], type=field.type) for field in PACKED_SCHEMA],
                    schema=PACKED_SCHEMA,
                ),
                self.output_path,
                compression=None if self.compression == "none" else self.compression,
            )
        else:
            self._writer.close()

    def abort(self) -> None:
        """Close a partially written shard after a processing failure."""
        if self._writer is not None:
            self._writer.close()


def realign_file(
    input_path: str,
    output_path: str,
    *,
    eos_id: int,
    pack_size: int,
    alignment: int,
    input_batch_rows: int,
    output_row_group_size: int,
    compression: str,
) -> FileStats:
    """Realign and repack one input Parquet shard."""
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    stats = FileStats(input_file=str(source), output_file=str(destination))

    current_ids: list[int] = []
    current_mask: list[int] = []
    current_starts: list[int] = []
    current_runtime_length = 0
    writer = PackedShardWriter(
        temporary,
        compression=compression,
        output_row_group_size=output_row_group_size,
    )

    def flush_pack() -> None:
        nonlocal current_ids, current_mask, current_starts, current_runtime_length
        if not current_starts:
            return
        if current_runtime_length > pack_size:
            raise AssertionError(f"Internal error: output pack has {current_runtime_length} > {pack_size} tokens")
        writer.append(current_ids, current_mask, current_starts)
        stats.output_rows += 1
        stats.output_sequences += len(current_starts)
        current_ids = []
        current_mask = []
        current_starts = []
        current_runtime_length = 0

    try:
        source_file = pq.ParquetFile(source)
        stats.input_rows = source_file.metadata.num_rows
        for sequence_ids, sequence_mask in iter_sequences(source, input_batch_rows=input_batch_rows):
            stats.input_sequences += 1
            runtime_length = len(sequence_ids) - 1
            if runtime_length == 0:
                stats.dropped_zero_runtime_sequences += 1
                continue
            if runtime_length > pack_size:
                raise ValueError(
                    f"{source}: sequence runtime length {runtime_length} exceeds pack_size {pack_size}; "
                    "truncate the source intentionally before running this tool"
                )

            target_stored_length = aligned_stored_length(len(sequence_ids), alignment)
            aligned_runtime_length = target_stored_length - 1
            if aligned_runtime_length > pack_size:
                # A source sequence with runtime length just below pack_size can round up exactly
                # to pack_size, but it must never round beyond it when pack_size is aligned.
                raise ValueError(
                    f"{source}: sequence runtime length {runtime_length} rounds to "
                    f"{aligned_runtime_length}, beyond pack_size {pack_size}"
                )
            if current_runtime_length + aligned_runtime_length > pack_size:
                flush_pack()

            pad_count = target_stored_length - len(sequence_ids)
            current_starts.append(len(current_ids))
            current_ids.extend(int(token) for token in sequence_ids)
            current_mask.extend(int(value) for value in sequence_mask)
            if pad_count:
                current_ids.extend([eos_id] * pad_count)
                current_mask.extend([0] * pad_count)
            current_runtime_length += aligned_runtime_length
            stats.original_runtime_tokens += runtime_length
            stats.padding_tokens_added += pad_count

        flush_pack()
        writer.close()
        os.replace(temporary, destination)
    except BaseException:
        writer.abort()
        temporary.unlink(missing_ok=True)
        raise

    return stats


def validate_file(path: Path, *, pack_size: int, alignment: int, input_batch_rows: int) -> tuple[int, int]:
    """Validate all subsequence boundaries in one output shard."""
    rows = pq.ParquetFile(path).metadata.num_rows
    sequences = 0
    runtime_in_pack = 0
    previous_row: int | None = None

    source = pq.ParquetFile(path)
    for batch_index, batch in enumerate(
        source.iter_batches(batch_size=input_batch_rows, columns=["input_ids", "loss_mask", "seq_start_id"])
    ):
        for row_index, (input_ids, loss_mask, starts) in enumerate(
            zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist())
        ):
            absolute_row = batch_index * input_batch_rows + row_index
            if len(input_ids) != len(loss_mask):
                raise ValueError(f"{path}: row {absolute_row} has mismatched input_ids/loss_mask lengths")
            boundaries = [*starts, len(input_ids)]
            runtime_in_pack = 0
            for start, end in zip(boundaries, boundaries[1:]):
                runtime_length = end - start - 1
                if runtime_length <= 0 or runtime_length % alignment != 0:
                    raise ValueError(
                        f"{path}: row {absolute_row} has runtime subsequence length "
                        f"{runtime_length}, which is not a positive multiple of {alignment}"
                    )
                runtime_in_pack += runtime_length
                sequences += 1
            if runtime_in_pack > pack_size:
                raise ValueError(f"{path}: row {absolute_row} contains {runtime_in_pack} > {pack_size} runtime tokens")
            previous_row = absolute_row

    if rows and previous_row is None:
        raise ValueError(f"{path}: Parquet metadata reports {rows} rows but no rows were readable")
    return rows, sequences


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing packed Parquet shards")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for aligned Parquet shards")
    parser.add_argument("--eos-id", type=int, required=True, help="EOS token id used for zero-loss alignment padding")
    parser.add_argument("--pack-size", type=int, default=262_144, help="Maximum runtime tokens per output pack")
    parser.add_argument("--context-parallel-size", type=int, required=True, help="Training CP size")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Training TP size")
    parser.add_argument(
        "--sequence-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the CP*TP sequence-parallel alignment constraint (default: enabled)",
    )
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=None,
        help="Optional extra alignment; combined with topology constraints using LCM",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--input-batch-rows", type=int, default=4)
    parser.add_argument("--output-row-group-size", type=int, default=16)
    parser.add_argument("--compression", choices=("zstd", "snappy", "gzip", "none"), default="zstd")
    parser.add_argument("--overwrite", action="store_true", help="Replace output shards that already exist")
    parser.add_argument("--resume", action="store_true", help="Skip output shards that already exist")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    """Run parallel shard-local realignment and write a summary manifest."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("output-dir must differ from input-dir; in-place rewriting is intentionally unsupported")
    if output_dir.is_relative_to(input_dir):
        raise ValueError("output-dir must not be nested under input-dir")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    for name in ("pack_size", "workers", "input_batch_rows", "output_row_group_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")

    alignment = required_alignment(
        context_parallel_size=args.context_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        sequence_parallel=args.sequence_parallel,
        explicit_multiple=args.pad_to_multiple,
    )
    if args.pack_size % alignment != 0:
        raise ValueError(f"pack_size {args.pack_size} must be divisible by required alignment {alignment}")

    input_files = sorted(input_dir.rglob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No .parquet files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, Path]] = []
    expected_outputs: list[Path] = []
    for source in input_files:
        destination = output_dir / source.relative_to(input_dir)
        expected_outputs.append(destination)
        if destination.exists():
            if args.resume:
                LOGGER.info("Skipping existing output shard: %s", destination)
                continue
            if not args.overwrite:
                raise FileExistsError(f"Output exists: {destination}; use --resume or --overwrite")
        tasks.append((source, destination))

    LOGGER.info(
        "Realigning %d shard(s): CP=%d TP=%d SP=%s alignment=%d pack_size=%d workers=%d",
        len(tasks),
        args.context_parallel_size,
        args.tensor_parallel_size,
        args.sequence_parallel,
        alignment,
        args.pack_size,
        args.workers,
    )

    completed: list[FileStats] = []
    worker_kwargs = {
        "eos_id": args.eos_id,
        "pack_size": args.pack_size,
        "alignment": alignment,
        "input_batch_rows": args.input_batch_rows,
        "output_row_group_size": args.output_row_group_size,
        "compression": args.compression,
    }
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(realign_file, str(source), str(destination), **worker_kwargs): (source, destination)
            for source, destination in tasks
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            stats = future.result()
            completed.append(stats)
            LOGGER.info(
                "[%d/%d] %s: %d sequences -> %d packs, added %d padding tokens",
                completed_count,
                len(tasks),
                Path(stats.input_file).name,
                stats.input_sequences,
                stats.output_rows,
                stats.padding_tokens_added,
            )

    if args.validate:
        LOGGER.info("Validating %d output shard(s)", len(expected_outputs))
        for output_path in expected_outputs:
            validate_file(
                output_path,
                pack_size=args.pack_size,
                alignment=alignment,
                input_batch_rows=args.input_batch_rows,
            )

    total = FileStats(input_file=str(input_dir), output_file=str(output_dir))
    for stats in completed:
        total.merge(stats)
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "eos_id": args.eos_id,
        "pack_size": args.pack_size,
        "context_parallel_size": args.context_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "sequence_parallel": args.sequence_parallel,
        "alignment": alignment,
        "stats": asdict(total),
        "files": [asdict(stats) for stats in sorted(completed, key=lambda item: item.input_file)],
    }
    manifest_path = output_dir / "realign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Done: %d sequences -> %d packs; added %d padding tokens. Manifest: %s",
        total.input_sequences,
        total.output_rows,
        total.padding_tokens_added,
        manifest_path,
    )


if __name__ == "__main__":
    main()
