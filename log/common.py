"""CSV, terminal logging, and timing helpers shared by CIFAR stages."""

from __future__ import annotations

import csv
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, TextIO, TypeVar

import torch


T = TypeVar("T")
Timings = dict[str, list[float]]
TIMING_FIELDS = ("stage", "calls", "total_seconds", "mean_seconds", "percentage")


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_with_log(log_path: Path, operation: Callable[[], None]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        stdout = TeeStream(sys.stdout, handle)
        stderr = TeeStream(sys.stderr, handle)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"run_log={log_path}")
            operation()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def save_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def measure(
    stage: str,
    device: torch.device,
    timings: Timings,
    operation: Callable[[], T],
    *,
    step: int | None = None,
) -> T:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    timings.setdefault(stage, []).append(elapsed)
    if step is None:
        print(f"[TIME] {stage}={elapsed:.3f}s")
    return result


def build_timing_rows(timings: Timings) -> list[dict[str, object]]:
    measured_total = sum(sum(values) for values in timings.values())
    return [
        {
            "stage": stage,
            "calls": len(values),
            "total_seconds": sum(values),
            "mean_seconds": sum(values) / len(values),
            "percentage": 100.0 * sum(values) / measured_total if measured_total else 0.0,
        }
        for stage, values in timings.items()
    ]


def print_timing_summary(timings: Timings) -> None:
    print("\n[TIMING SUMMARY]")
    for row in build_timing_rows(timings):
        print(
            f"{row['stage']:<30} total={row['total_seconds']:>10.3f} sec  "
            f"mean={row['mean_seconds']:>9.3f}  calls={row['calls']:>2}  "
            f"({row['percentage']:>6.2f}%)"
        )
    measured_total = sum(sum(values) for values in timings.values())
    print(f"{'measured_total':<30} total={measured_total:>10.3f} sec")
