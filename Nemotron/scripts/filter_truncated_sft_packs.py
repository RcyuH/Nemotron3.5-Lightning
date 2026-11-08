#!/usr/bin/env python3
"""Audit packed SFT splits and optionally remove truncation candidates.

The Nemotron SFT packer writes rows with this schema:

    input_ids: list<int32>
    loss_mask: list<uint8>
    seq_start_id: list<int32>

The packed output does not retain each sequence's length before packing. This
script prefers the sibling spool metadata for exact detection. It also
understands output from ``tools/realign_packed_sft_parquet.py`` when
``--realigned`` is passed: realigned
``pack_size`` limits runtime tokens, where runtime pack length is
``len(input_ids) - len(seq_start_id)``.

When exact spool metadata is unavailable, the conservative signature is a
full, single-sequence pack. For ordinary packed data that means:

    len(input_ids) == pack_size and seq_start_id == [0]

For realigned data, "full" refers to runtime length instead. These fallback
signatures can also match naturally exact-length sequences. Truncation
previously performed by ``max_doc_tokens`` cannot be attributed to individual
rows from the packed output alone.

By default this script only reports statistics.  Supplying ``--output-dir``
writes a new split tree and excludes every row matching the signature above.
The input files are never modified in place.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - only used in minimal environments
    tqdm = None

REQUIRED_COLUMNS = ("input_ids", "loss_mask", "seq_start_id")


@dataclass
class Counters:
    files: int = 0
    packs: int = 0
    sequences: int = 0
    tokens: int = 0
    trainable_tokens: int = 0
    full_packs: int = 0
    full_multi_sequence_packs: int = 0
    truncation_candidate_packs: int = 0
    truncation_candidate_tokens: int = 0
    zero_trainable_packs: int = 0
    zero_trainable_sequences: int = 0
    kept_packs: int = 0
    dropped_packs: int = 0
    exact_detection_files: int = 0
    heuristic_detection_files: int = 0
    max_doc_tokens_truncations_reported: int = 0
    loss_mask_scanned_files: int = 0
    linked_unchanged_files: int = 0
    content_stats_skipped_files: int = 0

    def add(self, other: Counters) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_report(self) -> dict[str, Any]:
        result = asdict(self)
        result["truncation_candidate_pack_rate"] = self.truncation_candidate_packs / self.packs if self.packs else 0.0
        result["zero_trainable_pack_rate"] = self.zero_trainable_packs / self.packs if self.packs else 0.0
        return result


@dataclass(frozen=True)
class DetectionPlan:
    mode: str
    candidate_rows: list[bool] | None
    spool_dir: str | None
    runtime_packing: bool = False
    source_path: str | None = None
    max_doc_tokens: int | None = None
    max_doc_tokens_truncations_reported: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Nemotron packed-SFT split Parquets and optionally write a copy "
            "without full, single-sequence packs (pack-size truncation candidates)."
        )
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Folder containing train/valid/test directories (searched recursively for *.parquet).",
    )
    parser.add_argument(
        "--pack-size",
        type=int,
        required=True,
        help="Exact pack_size used by SFT data preparation, for example 4096.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write cleaned Parquets here. Omit for report-only mode.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional report path. Defaults to OUTPUT_DIR/truncation_report.json when filtering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Parquet rows read per batch (default: 16; suitable for 256K packs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Number of Parquet shards processed in parallel (default: min(16, CPU count)).",
    )
    parser.add_argument("--compression", default="zstd", help="Compression for cleaned Parquets.")
    parser.add_argument(
        "--scan-loss-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also materialize loss_mask to count trainable/zero-loss tokens. This is much slower for long contexts "
            "and is disabled by default."
        ),
    )
    parser.add_argument(
        "--link-unchanged",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For exact detection, create a symlink instead of rewriting shards with no truncated packs. "
            "This is much faster but the cleaned tree depends on the source files."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (default: enabled).",
    )
    parser.add_argument(
        "--realigned",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Input was produced by realign_packed_sft_parquet.py. Interpret pack_size as runtime tokens "
            "and map rows through realign_manifest.json to the source spool. Default: normal packed data."
        ),
    )
    parser.add_argument(
        "--require-spool",
        action="store_true",
        help="Fail instead of using the conservative packed-Parquet heuristic when spool metadata is unavailable.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.pack_size <= 0:
        raise ValueError(f"--pack-size must be positive, got {args.pack_size}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")

    splits_dir = args.splits_dir.expanduser().resolve()
    if not splits_dir.is_dir():
        raise ValueError(f"--splits-dir is not a directory: {splits_dir}")

    output_dir: Path | None = None
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir == splits_dir or splits_dir in output_dir.parents:
            raise ValueError("--output-dir must not be the input directory or a child of it")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"--output-dir must be absent or empty: {output_dir}")

    return splits_dir, output_dir


def validate_schema(path: Path, schema: pa.Schema) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}; schema={schema}")


def sequence_trainable_counts(loss_mask: list[int], starts: list[int]) -> list[int]:
    if not starts:
        return []
    ends = starts[1:] + [len(loss_mask)]
    return [sum(int(x) for x in loss_mask[start:end]) for start, end in zip(starts, ends, strict=True)]


def reconstruct_bins(lengths: list[int], pack_size: int, algorithm: str, seed: int | None) -> list[list[int]]:
    indexed_lengths = [(min(length, pack_size), index) for index, length in enumerate(lengths)]

    if algorithm == "first_fit_decreasing":
        indexed_lengths.sort(key=lambda item: item[0], reverse=True)
    elif algorithm == "first_fit_shuffle":
        random.Random(seed).shuffle(indexed_lengths)
    elif algorithm == "concatenative":
        bins: list[list[int]] = [[]]
        current_size = 0
        for length, index in indexed_lengths:
            if current_size + length <= pack_size:
                bins[-1].append(index)
                current_size += length
            else:
                bins.append([index])
                current_size = length
        return [item for item in bins if item]
    else:
        raise ValueError(f"Unsupported packing algorithm in spool manifest: {algorithm!r}")

    bins = []
    remaining: list[int] = []
    for length, index in indexed_lengths:
        for bin_index, capacity in enumerate(remaining):
            if capacity >= length:
                bins[bin_index].append(index)
                remaining[bin_index] -= length
                break
        else:
            bins.append([index])
            remaining.append(pack_size - length)
    return bins


def load_spool(spool_dir: Path) -> tuple[dict[str, Any], list[int]]:
    manifest_path = spool_dir / "manifest.json"
    lengths_path = spool_dir / "lengths.bin"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lengths_array = array("I")
    if lengths_array.itemsize != 4:
        raise RuntimeError(f"This platform uses {lengths_array.itemsize}-byte unsigned ints; expected 4")
    lengths_bytes = lengths_path.stat().st_size
    if lengths_bytes % lengths_array.itemsize:
        raise ValueError(f"{lengths_path}: byte size {lengths_bytes} is not divisible by 4")
    with lengths_path.open("rb") as lengths_file:
        lengths_array.fromfile(lengths_file, lengths_bytes // lengths_array.itemsize)
    lengths = [int(length) for length in lengths_array]
    expected_sequences = int(manifest.get("num_sequences", len(lengths)))
    if len(lengths) != expected_sequences:
        raise ValueError(f"{lengths_path}: found {len(lengths)} lengths, manifest says {expected_sequences} sequences")
    return manifest, lengths


def spool_dir_for_source(source_path: Path) -> Path:
    resolved_path = source_path.resolve()
    return resolved_path.parent / "spool" / resolved_path.stem


def exact_plan_from_source_spool(
    *,
    source_path: Path,
    output_path: Path,
    requested_pack_size: int,
    realigned: bool,
) -> DetectionPlan | None:
    spool_dir = spool_dir_for_source(source_path)
    manifest_path = spool_dir / "manifest.json"
    lengths_path = spool_dir / "lengths.bin"
    if not manifest_path.is_file() or not lengths_path.is_file():
        return None

    manifest, lengths = load_spool(spool_dir)
    source_pack_size = int(manifest.get("pack_size", 0) or 0)
    if source_pack_size <= 0:
        raise ValueError(f"{manifest_path}: invalid pack_size={source_pack_size}")
    if not realigned and source_pack_size != requested_pack_size:
        raise ValueError(
            f"{manifest_path}: manifest pack_size={source_pack_size}, but --pack-size={requested_pack_size}"
        )

    algorithm = str(manifest.get("algorithm", ""))
    seed_value = manifest.get("seed")
    seed = None if seed_value is None else int(seed_value)
    bins = reconstruct_bins(lengths, source_pack_size, algorithm, seed)

    if realigned:
        # realign_packed_sft_parquet iterates source packed rows in order and
        # preserves logical sequence order while changing row boundaries.
        candidate_sequences: list[bool] = []
        for indices in bins:
            for index in indices:
                stored_length = min(lengths[index], source_pack_size)
                if stored_length - 1 == 0:  # dropped by the realigner
                    continue
                candidate_sequences.append(lengths[index] > source_pack_size)

        candidate_rows = []
        sequence_offset = 0
        output_file = pq.ParquetFile(output_path)
        for batch in output_file.iter_batches(columns=["seq_start_id"]):
            for starts in batch.column(0).to_pylist():
                count = len(starts)
                flags = candidate_sequences[sequence_offset : sequence_offset + count]
                if len(flags) != count:
                    raise ValueError(f"{output_path}: output contains more sequences than its realign source")
                candidate_rows.append(any(flags))
                sequence_offset += count
        if sequence_offset != len(candidate_sequences):
            raise ValueError(
                f"{output_path}: mapped {sequence_offset} output sequences, "
                f"but source contains {len(candidate_sequences)} non-empty sequences"
            )
        mode = "exact_realign_source_spool"
    else:
        candidate_rows = [any(lengths[index] > source_pack_size for index in indices) for indices in bins]
        mode = "exact_spool"

    tokenization_stats = manifest.get("tokenization_stats", {}) or {}
    max_doc_tokens = manifest.get("max_doc_tokens")
    return DetectionPlan(
        mode=mode,
        candidate_rows=candidate_rows,
        spool_dir=str(spool_dir),
        runtime_packing=realigned,
        source_path=str(source_path),
        max_doc_tokens=None if max_doc_tokens is None else int(max_doc_tokens),
        max_doc_tokens_truncations_reported=int(tokenization_stats.get("num_truncated", 0) or 0),
    )


def load_detection_plan(
    input_path: Path,
    relative_path: Path,
    pack_size: int,
    require_spool: bool,
    realigned: bool,
    realign_source_dir: Path | None,
) -> DetectionPlan:
    if realigned:
        if realign_source_dir is None:
            raise ValueError(
                "--realigned requires a valid realign_manifest.json in --splits-dir "
                "(or its parent) so aligned rows can be mapped to source sequences"
            )
        source_path = realign_source_dir / relative_path
        if not source_path.is_file():
            raise ValueError(f"Realign source shard not found for {input_path}: {source_path}")
        plan = exact_plan_from_source_spool(
            source_path=source_path,
            output_path=input_path,
            requested_pack_size=pack_size,
            realigned=True,
        )
        if plan is not None:
            return plan
        expected_spool = spool_dir_for_source(source_path)
        if require_spool:
            raise ValueError(f"Source spool metadata not found for {input_path}; expected {expected_spool}")
        return DetectionPlan(
            mode="heuristic_realign",
            candidate_rows=None,
            spool_dir=None,
            runtime_packing=True,
            source_path=str(source_path),
        )

    plan = exact_plan_from_source_spool(
        source_path=input_path,
        output_path=input_path,
        requested_pack_size=pack_size,
        realigned=False,
    )
    if plan is not None:
        return plan
    expected_spool = spool_dir_for_source(input_path)
    if require_spool:
        raise ValueError(f"Spool metadata not found for {input_path}; expected {expected_spool}")
    return DetectionPlan(mode="heuristic", candidate_rows=None, spool_dir=None)


def inspect_batch(
    batch: pa.RecordBatch,
    pack_size: int,
    exact_candidates: list[bool] | None = None,
    runtime_packing: bool = False,
    scan_loss_mask: bool = False,
) -> tuple[list[bool], Counters]:
    ids_column = batch.column(batch.schema.get_field_index("input_ids"))
    mask_column = batch.column(batch.schema.get_field_index("loss_mask"))
    ids_lengths = pc.list_value_length(ids_column).to_pylist()
    mask_lengths = pc.list_value_length(mask_column).to_pylist()
    start_rows = batch.column(batch.schema.get_field_index("seq_start_id")).to_pylist()
    mask_rows = mask_column.to_pylist() if scan_loss_mask else None

    keep: list[bool] = []
    stats = Counters()

    if exact_candidates is not None and len(exact_candidates) != batch.num_rows:
        raise ValueError(
            f"Exact candidate slice has {len(exact_candidates)} rows, but Parquet batch has {batch.num_rows}"
        )

    for row_index, (pack_tokens, mask_tokens, starts) in enumerate(
        zip(ids_lengths, mask_lengths, start_rows, strict=True)
    ):
        if pack_tokens is None or mask_tokens is None or starts is None:
            raise ValueError(f"Null packed value at batch row {row_index}")
        if pack_tokens != mask_tokens:
            raise ValueError(
                f"input_ids/loss_mask length mismatch at batch row {row_index}: {pack_tokens} != {mask_tokens}"
            )
        if not starts or starts[0] != 0 or any(a >= b for a, b in zip(starts, starts[1:])):
            raise ValueError(f"Invalid seq_start_id at batch row {row_index}: {starts}")
        if any(start < 0 or start >= pack_tokens for start in starts):
            raise ValueError(
                f"seq_start_id outside input_ids at batch row {row_index}: starts={starts}, len={pack_tokens}"
            )

        runtime_pack_tokens = pack_tokens - len(starts)
        measured_pack_tokens = runtime_pack_tokens if runtime_packing else pack_tokens
        if measured_pack_tokens > pack_size:
            mode_hint = "" if runtime_packing else "; use --realigned if this data came from the realign tool"
            raise ValueError(
                f"{'Runtime' if runtime_packing else 'Stored'} pack length {measured_pack_tokens} exceeds "
                f"--pack-size={pack_size} at batch row {row_index} "
                f"(stored={pack_tokens}, runtime={runtime_pack_tokens}, sequences={len(starts)}){mode_hint}"
            )
        if mask_rows is not None:
            trainable_per_sequence = sequence_trainable_counts(mask_rows[row_index], starts)
            trainable_tokens = sum(trainable_per_sequence)
        else:
            trainable_per_sequence = []
            trainable_tokens = 0
        is_full = measured_pack_tokens == pack_size
        is_candidate = bool(exact_candidates[row_index]) if exact_candidates is not None else is_full and starts == [0]

        stats.packs += 1
        stats.sequences += len(starts)
        stats.tokens += pack_tokens
        stats.trainable_tokens += trainable_tokens
        stats.full_packs += int(is_full)
        stats.full_multi_sequence_packs += int(is_full and len(starts) > 1)
        stats.truncation_candidate_packs += int(is_candidate)
        stats.truncation_candidate_tokens += pack_tokens if is_candidate else 0
        if mask_rows is not None:
            stats.zero_trainable_packs += int(trainable_tokens == 0)
            stats.zero_trainable_sequences += sum(count == 0 for count in trainable_per_sequence)
        stats.kept_packs += int(not is_candidate)
        stats.dropped_packs += int(is_candidate)
        keep.append(not is_candidate)

    return keep, stats


def process_file(
    input_path: Path,
    output_path: Path | None,
    *,
    pack_size: int,
    batch_size: int,
    compression: str,
    require_spool: bool,
    relative_path: Path,
    realigned: bool,
    realign_source_dir: Path | None,
    show_progress: bool,
    scan_loss_mask: bool,
    link_unchanged: bool,
) -> tuple[Counters, DetectionPlan]:
    parquet_file = pq.ParquetFile(input_path)
    validate_schema(input_path, parquet_file.schema_arrow)
    detection = load_detection_plan(
        input_path,
        relative_path,
        pack_size,
        require_spool,
        realigned,
        realign_source_dir,
    )
    if detection.candidate_rows is not None and len(detection.candidate_rows) != parquet_file.metadata.num_rows:
        raise ValueError(
            f"{input_path}: reconstructed {len(detection.candidate_rows)} bins from spool, "
            f"but Parquet contains {parquet_file.metadata.num_rows} rows"
        )

    writer: pq.ParquetWriter | None = None
    tmp_path: Path | None = None
    should_link = (
        output_path is not None
        and link_unchanged
        and detection.candidate_rows is not None
        and not any(detection.candidate_rows)
    )

    if should_link:
        assert output_path is not None
        sequence_count = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["seq_start_id"]):
            sequence_count += sum(len(starts) for starts in batch.column(0).to_pylist())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_link = output_path.with_name(output_path.name + ".tmp-link")
        relative_target = os.path.relpath(input_path.resolve(), output_path.parent)
        temporary_link.symlink_to(relative_target)
        os.replace(temporary_link, output_path)
        return (
            Counters(
                files=1,
                packs=parquet_file.metadata.num_rows,
                sequences=sequence_count,
                kept_packs=parquet_file.metadata.num_rows,
                exact_detection_files=int(detection.mode.startswith("exact_")),
                heuristic_detection_files=int(detection.mode.startswith("heuristic")),
                max_doc_tokens_truncations_reported=detection.max_doc_tokens_truncations_reported,
                linked_unchanged_files=1,
                content_stats_skipped_files=1,
            ),
            detection,
        )

    if output_path is not None and not should_link:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_name(output_path.name + ".tmp")
        writer = pq.ParquetWriter(tmp_path, parquet_file.schema_arrow, compression=compression)

    stats = Counters(files=1)
    stats.exact_detection_files = int(detection.mode.startswith("exact_"))
    stats.heuristic_detection_files = int(detection.mode.startswith("heuristic"))
    stats.max_doc_tokens_truncations_reported = detection.max_doc_tokens_truncations_reported
    stats.loss_mask_scanned_files = int(scan_loss_mask)
    stats.linked_unchanged_files = int(should_link)
    row_offset = 0
    try:
        batches = parquet_file.iter_batches(batch_size=batch_size)
        if tqdm is not None and show_progress:
            total_batches = (parquet_file.metadata.num_rows + batch_size - 1) // batch_size
            batches = tqdm(
                batches,
                total=total_batches,
                desc=f"  {relative_path}",
                unit="batch",
                leave=False,
            )
        for batch in batches:
            exact_slice = None
            if detection.candidate_rows is not None:
                exact_slice = detection.candidate_rows[row_offset : row_offset + batch.num_rows]
            keep, batch_stats = inspect_batch(
                batch,
                pack_size,
                exact_slice,
                runtime_packing=detection.runtime_packing,
                scan_loss_mask=scan_loss_mask,
            )
            row_offset += batch.num_rows
            stats.add(batch_stats)
            if writer is not None:
                filtered = pa.Table.from_batches([batch]).filter(pa.array(keep, type=pa.bool_()))
                if filtered.num_rows:
                    writer.write_table(filtered)
    except Exception:
        if writer is not None:
            writer.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()
            assert tmp_path is not None and output_path is not None
            os.replace(tmp_path, output_path)

    return stats, detection


def split_name(relative_path: Path) -> str:
    return relative_path.parts[0] if len(relative_path.parts) > 1 else "root"


def discover_realign_source(
    splits_dir: Path,
    pack_size: int,
    realigned: bool,
) -> tuple[Path | None, Path | None]:
    if not realigned:
        return None, None
    candidates = [splits_dir / "realign_manifest.json", splits_dir.parent / "realign_manifest.json"]
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_output = Path(str(manifest.get("output_dir", ""))).expanduser().resolve()
        if manifest_output != splits_dir:
            continue
        manifest_pack_size = int(manifest.get("pack_size", 0) or 0)
        if manifest_pack_size != pack_size:
            raise ValueError(f"{manifest_path}: realign pack_size={manifest_pack_size}, but --pack-size={pack_size}")
        source_dir = Path(str(manifest.get("input_dir", ""))).expanduser().resolve()
        if not source_dir.is_dir():
            raise ValueError(f"{manifest_path}: realign input_dir no longer exists: {source_dir}")
        return source_dir, manifest_path
    return None, None


def main() -> int:
    args = parse_args()
    try:
        splits_dir, output_dir = validate_args(args)
        parquet_paths = sorted(path for path in splits_dir.rglob("*.parquet") if path.is_file())
        if not parquet_paths:
            raise ValueError(f"No *.parquet files found under {splits_dir}")
        realign_source_dir, realign_manifest_path = discover_realign_source(
            splits_dir,
            args.pack_size,
            args.realigned,
        )
        if args.realigned and realign_manifest_path is None:
            raise ValueError(
                "--realigned was set, but no matching realign_manifest.json was found in --splits-dir or its parent"
            )

        total = Counters()
        by_split: dict[str, Counters] = {}
        by_file: dict[str, dict[str, Any]] = {}

        if args.progress and tqdm is None:
            print("warning: tqdm is not installed; continuing without progress bars", file=sys.stderr)

        def record_result(
            relative_path: Path,
            file_stats: Counters,
            detection: DetectionPlan,
        ) -> None:
            total.add(file_stats)
            split = split_name(relative_path)
            by_split.setdefault(split, Counters()).add(file_stats)
            file_report = file_stats.to_report()
            file_report["detection_mode"] = detection.mode
            file_report["spool_dir"] = detection.spool_dir
            file_report["source_path"] = detection.source_path
            file_report["runtime_packing"] = detection.runtime_packing
            file_report["max_doc_tokens"] = detection.max_doc_tokens
            by_file[str(relative_path)] = file_report

        common_kwargs = {
            "pack_size": args.pack_size,
            "batch_size": args.batch_size,
            "compression": args.compression,
            "require_spool": args.require_spool,
            "realigned": args.realigned,
            "realign_source_dir": realign_source_dir,
            "scan_loss_mask": args.scan_loss_mask,
            "link_unchanged": args.link_unchanged,
        }

        if args.workers == 1 or len(parquet_paths) == 1:
            shard_iterator = parquet_paths
            if tqdm is not None and args.progress:
                shard_iterator = tqdm(parquet_paths, desc="Parquet shards", unit="shard")
            for input_path in shard_iterator:
                relative_path = input_path.relative_to(splits_dir)
                output_path = output_dir / relative_path if output_dir is not None else None
                file_stats, detection = process_file(
                    input_path,
                    output_path,
                    relative_path=relative_path,
                    show_progress=args.progress,
                    **common_kwargs,
                )
                record_result(relative_path, file_stats, detection)
        else:
            max_workers = min(args.workers, len(parquet_paths))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for input_path in parquet_paths:
                    relative_path = input_path.relative_to(splits_dir)
                    output_path = output_dir / relative_path if output_dir is not None else None
                    future = executor.submit(
                        process_file,
                        input_path,
                        output_path,
                        relative_path=relative_path,
                        show_progress=False,
                        **common_kwargs,
                    )
                    futures[future] = relative_path

                completed = as_completed(futures)
                if tqdm is not None and args.progress:
                    completed = tqdm(completed, total=len(futures), desc="Parquet shards", unit="shard")
                for future in completed:
                    relative_path = futures[future]
                    file_stats, detection = future.result()
                    record_result(relative_path, file_stats, detection)

        report = {
            "input_splits_dir": str(splits_dir),
            "output_splits_dir": str(output_dir) if output_dir is not None else None,
            "realign_manifest": str(realign_manifest_path) if realign_manifest_path is not None else None,
            "realign_source_dir": str(realign_source_dir) if realign_source_dir is not None else None,
            "pack_size": args.pack_size,
            "input_mode": "realigned_runtime_packing" if args.realigned else "normal_stored_packing",
            "workers": args.workers,
            "batch_size": args.batch_size,
            "scan_loss_mask": args.scan_loss_mask,
            "link_unchanged": args.link_unchanged,
            "mode": "filter" if output_dir is not None else "report_only",
            "detection": {
                "exact": (
                    "reconstruct source bins from spool/lengths.bin; for realigned data, map source sequence "
                    "order onto new packed rows"
                ),
                "fallback_signature": "full runtime/stored pack with seq_start_id == [0]",
                "guarantee": "exact when spool metadata is available; conservative fallback otherwise",
                "limitation": (
                    "fallback may include naturally exact-length sequences; max_doc_tokens truncations cannot "
                    "be attributed to individual packed rows because spool lengths are already truncated"
                ),
            },
            "total": total.to_report(),
            "splits": {name: stats.to_report() for name, stats in sorted(by_split.items())},
            "files": by_file,
        }

        report_text = json.dumps(report, indent=2, sort_keys=True)
        print(report_text)

        report_path = args.report_json
        if report_path is None and output_dir is not None:
            report_path = output_dir / "truncation_report.json"
        if report_path is not None:
            report_path = report_path.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report_text + "\n", encoding="utf-8")

        return 0
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
