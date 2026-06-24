"""Scrub a halted agentic-baseline JSONL log so a subsequent --resume retries
the silent-fail records instead of treating them as done.

Conservative drop criterion (from adversarial review). Drop a record iff ALL of:
  - error is not None
  - predicted_codes is empty
  - predicted_codes_invalid is empty
  - raw_codes_field is empty
  - cost_usd == 0
  - num_turns == 0

Anything that produced ANY signal (codes, invalid codes, raw codes_field, paid
tokens, or completed turns) is kept. This avoids silently deleting "answered
but wrong" data points on future halted batches.

Atomic procedure:
  1. Acquire <jsonl>.lock via O_CREAT|O_EXCL; abort if held.
  2. Snapshot mtime of <jsonl>.
  3. Backup <jsonl> -> <jsonl>.pre_scrub.bak (via .bak.tmp + os.replace).
  4. Stream <jsonl> line-by-line into <jsonl>.scrub.tmp; apply keep predicate;
     dedupe by parquet_idx keeping the record with non-empty predicted_codes
     if any duplicate, else the last one.
  5. Verify mtime unchanged; abort + rollback if changed.
  6. os.replace(.scrub.tmp, <jsonl>).
  7. Write <jsonl-base>.scrub_audit.json via tmp+replace recording the dropped
     count, reason buckets, and per-source drop counts.
  8. Regenerate <jsonl-base>.results.json snapshot from the cleaned data via
     save_results_snapshot (preserved fields: scrubbed_at, dropped_count,
     kept_count, original_total, original_cost_usd).
  9. Release lock.

Usage:
    python -m experiments.agentic_baseline_v2.scrub <jsonl-path>
    python -m experiments.agentic_baseline_v2.scrub <jsonl-path> --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.agentic_baseline_v2.run import (
    RecordResult,
    compute_metrics,
    load_jsonl_predictions,
    save_results_snapshot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("agentic_baseline_v2.scrub")


# ---------------------------------------------------------------------------
# Keep / drop predicate
# ---------------------------------------------------------------------------


def should_drop(record: dict[str, Any]) -> bool:
    """Return True iff the record produced no signal at all and should be
    dropped from the JSONL so resume retries the parquet_idx."""
    if not record.get("error"):
        return False  # success, never drop
    if record.get("predicted_codes"):
        return False  # valid prediction present, keep
    if record.get("predicted_codes_invalid"):
        return False  # produced shape-invalid codes; counts toward invalid_code_record_rate
    if record.get("raw_codes_field"):
        return False  # produced raw codes that failed downstream; signal exists
    if float(record.get("cost_usd") or 0.0) > 0:
        return False  # paid tokens were spent; subprocess produced real output
    if int(record.get("num_turns") or 0) > 0:
        return False  # CLI completed turns; not a silent fail
    return True


# ---------------------------------------------------------------------------
# Atomic file utilities
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy via tmp+rename so the destination only ever sees a complete file."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as r, tmp.open("wb") as w:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            w.write(chunk)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl_path", type=Path, help="Path to predictions.jsonl to scrub.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report drop/keep counts without rewriting any file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jsonl_path: Path = args.jsonl_path.resolve()
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} does not exist.", file=sys.stderr)
        return 2
    if jsonl_path.suffix != ".jsonl":
        print(f"WARN: expected .jsonl suffix, got {jsonl_path.suffix}.", file=sys.stderr)

    base_label = jsonl_path.name[: -len(".predictions.jsonl")] if jsonl_path.name.endswith(".predictions.jsonl") else jsonl_path.stem
    results_dir = jsonl_path.parent
    snapshot_path = results_dir / f"{base_label}.results.json"
    audit_path = results_dir / f"{base_label}.scrub_audit.json"
    lock_path = jsonl_path.with_suffix(jsonl_path.suffix + ".lock")
    pre_scrub_bak = results_dir / f"{base_label}.predictions.pre_scrub.bak.jsonl"
    pre_scrub_results_bak = results_dir / f"{base_label}.pre_scrub.results.json.bak"

    logger.info("JSONL:    %s", jsonl_path)
    logger.info("Snapshot: %s", snapshot_path)
    logger.info("Audit:    %s", audit_path)
    logger.info("Dry run:  %s", args.dry_run)

    # ---- Pre-flight scan to compute keep/drop counts and reason buckets.
    kept = 0
    dropped = 0
    drop_reason_buckets: Counter = Counter()
    drop_source_buckets: Counter = Counter()
    duplicate_parquet_idx_count = 0
    seen_idx: set[int] = set()
    total_cost_in_dropped = 0.0
    total_elapsed_in_dropped = 0.0
    original_total = 0
    original_cost_usd = 0.0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            original_total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                # Treat un-parseable lines as drops (they cannot be resumed safely).
                dropped += 1
                drop_reason_buckets["unparseable_jsonl_line"] += 1
                continue
            idx = d.get("parquet_idx")
            if idx in seen_idx:
                duplicate_parquet_idx_count += 1
            seen_idx.add(idx)
            original_cost_usd += float(d.get("cost_usd") or 0.0)

            if should_drop(d):
                dropped += 1
                err = (d.get("error") or "").strip().lower()
                if "claude exit 1" in err:
                    drop_reason_buckets["claude_exit_1_empty_stderr"] += 1
                elif "timeout" in err:
                    drop_reason_buckets["timeout"] += 1
                elif "json" in err:
                    drop_reason_buckets["unparseable_response"] += 1
                elif "agent reported error" in err:
                    drop_reason_buckets["agent_reported_error_no_signal"] += 1
                else:
                    drop_reason_buckets["other_silent_fail"] += 1
                drop_source_buckets[d.get("source_id", "")] += 1
                total_cost_in_dropped += float(d.get("cost_usd") or 0.0)
                total_elapsed_in_dropped += float(d.get("elapsed_s") or 0.0)
            else:
                kept += 1

    logger.info("Pre-flight: total=%d  keep=%d  drop=%d  duplicate_parquet_idx=%d  original_cost=$%.2f",
                original_total, kept, dropped, duplicate_parquet_idx_count, original_cost_usd)
    logger.info("Drop reasons: %s", dict(drop_reason_buckets))
    logger.info("Drop per source: %s", dict(drop_source_buckets))

    if args.dry_run:
        logger.info("Dry run: no files modified.")
        return 0

    # ---- Acquire lockfile (O_EXCL).
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"ERROR: lock file {lock_path} already exists. Another scrub or "
              f"a live run.py may be in progress. If you are SURE no other process "
              f"is touching the file, delete the lock and retry.", file=sys.stderr)
        return 3

    try:
        os.write(lock_fd, f"scrub pid={os.getpid()} at {dt.datetime.now().isoformat()}\n".encode())
        os.close(lock_fd)

        # ---- mtime snapshot.
        mtime0 = jsonl_path.stat().st_mtime

        # ---- Backup the original JSONL atomically.
        logger.info("Backing up to %s ...", pre_scrub_bak.name)
        _atomic_copy(jsonl_path, pre_scrub_bak)

        # ---- Backup the existing snapshot, if any.
        if snapshot_path.exists():
            logger.info("Backing up existing snapshot to %s ...", pre_scrub_results_bak.name)
            _atomic_copy(snapshot_path, pre_scrub_results_bak)

        # ---- Stream-rewrite into .scrub.tmp with dedupe by parquet_idx.
        scrub_tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".scrub.tmp")
        # Dedupe map: parquet_idx -> chosen record dict.
        # Rule: prefer record with non-empty predicted_codes; otherwise prefer
        # the one with higher cost_usd; otherwise keep the last one.
        chosen: dict[int, dict[str, Any]] = {}
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if should_drop(d):
                    continue
                idx = int(d.get("parquet_idx", -1))
                if idx not in chosen:
                    chosen[idx] = d
                    continue
                prev = chosen[idx]
                prev_has_valid = bool(prev.get("predicted_codes"))
                this_has_valid = bool(d.get("predicted_codes"))
                if this_has_valid and not prev_has_valid:
                    chosen[idx] = d
                elif this_has_valid == prev_has_valid:
                    if float(d.get("cost_usd") or 0.0) > float(prev.get("cost_usd") or 0.0):
                        chosen[idx] = d
                    # else keep prev (= earlier line, no improvement)
                # else keep prev (it has valid codes and this one does not)

        with scrub_tmp.open("w", encoding="utf-8") as w:
            for idx in sorted(chosen.keys()):
                w.write(json.dumps(chosen[idx], ensure_ascii=False) + "\n")

        # ---- Recheck mtime; abort + rollback if changed.
        mtime1 = jsonl_path.stat().st_mtime
        if mtime1 != mtime0:
            logger.error("JSONL mtime changed during scrub (%s -> %s); rolling back.",
                         mtime0, mtime1)
            try:
                scrub_tmp.unlink()
            except FileNotFoundError:
                pass
            return 4

        # ---- Atomic replace.
        os.replace(scrub_tmp, jsonl_path)
        kept_actual = len(chosen)
        logger.info("Replaced JSONL with cleaned copy: kept=%d unique parquet_idx", kept_actual)

        # ---- Write the audit JSON.
        audit_payload = {
            "scrubbed_at": dt.datetime.now().isoformat(),
            "jsonl_path": str(jsonl_path),
            "pre_scrub_backup": str(pre_scrub_bak),
            "original_total_lines": original_total,
            "original_cost_usd": round(original_cost_usd, 4),
            "kept_count": kept_actual,
            "dropped_count": original_total - kept_actual,
            "dropped_total_elapsed_s": round(total_elapsed_in_dropped, 1),
            "dropped_total_cost_usd": round(total_cost_in_dropped, 4),
            "duplicate_parquet_idx_in_original": duplicate_parquet_idx_count,
            "drop_reason_buckets": dict(drop_reason_buckets),
            "drop_source_buckets": dict(drop_source_buckets),
            "drop_criterion": (
                "error is not None AND empty predicted_codes AND empty "
                "predicted_codes_invalid AND empty raw_codes_field AND cost_usd==0 AND num_turns==0"
            ),
        }
        _atomic_write_json(audit_path, audit_payload)
        logger.info("Wrote audit -> %s", audit_path)

        # ---- Regenerate the snapshot from the cleaned JSONL.
        cleaned_records: list[RecordResult] = load_jsonl_predictions(jsonl_path)
        save_results_snapshot(cleaned_records, snapshot_path)
        # Add scrub metadata to the snapshot so a future reader can tell
        # this file was scrubbed and from what.
        snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snap["scrub_metadata"] = {
            "scrubbed_at": audit_payload["scrubbed_at"],
            "dropped_count": audit_payload["dropped_count"],
            "kept_count": audit_payload["kept_count"],
            "original_total": original_total,
            "original_cost_usd": audit_payload["original_cost_usd"],
            "audit_file": str(audit_path),
        }
        _atomic_write_json(snapshot_path, snap)
        logger.info("Regenerated snapshot -> %s", snapshot_path)

        # ---- Headline.
        m = snap.get("metrics", {})
        logger.info("HEADLINE on cleaned set: n=%s  macro_f1=%.4f  sample_f1=%.4f  exact=%.4f",
                    m.get("n"), m.get("macro_f1", 0.0), m.get("sample_f1", 0.0),
                    m.get("exact_set_match_rate", 0.0))
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
