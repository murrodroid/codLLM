#!/usr/bin/env python3
"""
make_gantt.py

Creates a Gantt chart PNG from the provided Task/Start/End/Duration table.

Usage:
  uv run make_gantt.py
  # or: python make_gantt.py
Outputs:
  gantt.png
"""

from __future__ import annotations

import re
from datetime import datetime
from io import StringIO

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


TSV_DATA = """Task\tStart\tEnd\tDuration
Infrastructure finalization (HPC, W&B, reproducibility)\tfeb. 25\tmar. 10\t13
Baseline implementation & evaluation protocol\tmar. 1\tmar. 20\t19
Structured-output LLM implementation\tmar. 10\tmar. 31\t21
Hierarchical multi-label model implementation\tmar. 15\tapr. 5\t21
Initial training runs (both models)\tmar. 25\tapr. 10\t16
Full experiments & hyperparameter tuning\tapr. 5\tapr. 25\t20
Quantitative & qualitative analysis\tapr. 15\tapr. 30\t15
Conference slide preparation\tapr. 25\t08-05-2026\t13
Conference presentation (May 10–13)\t10-05-2026\t13-05-2026\t3
Methods writing\tmar. 10\tapr. 15\t36
Results writing\tapr. 20\t20-05-2026\t30
Discussion writing\t15-05-2026\t05-06-2026\t21
Conclusion writing\t01-06-2026\t10-06-2026\t9
Full draft completion\t10-06-2026\t10-06-2026\t0
Revision & proofreading\t10-06-2026\t20-06-2026\t10
Final submission (report)\t20-06-2026\t20-06-2026\t0
Oral defense\t25-06-2026\t25-06-2026\t0
"""

DEFAULT_YEAR = 2026

MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "jan.": 1,
    "feb": 2,
    "february": 2,
    "feb.": 2,
    "mar": 3,
    "march": 3,
    "mar.": 3,
    "apr": 4,
    "april": 4,
    "apr.": 4,
    "may": 5,
    "may.": 5,
    "jun": 6,
    "june": 6,
    "jun.": 6,
    "jul": 7,
    "july": 7,
    "jul.": 7,
    "aug": 8,
    "august": 8,
    "aug.": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "sep.": 9,
    "sept.": 9,
    "oct": 10,
    "october": 10,
    "oct.": 10,
    "nov": 11,
    "november": 11,
    "nov.": 11,
    "dec": 12,
    "december": 12,
    "dec.": 12,
}


def parse_mixed_date(s: str) -> datetime:
    """Parse a variety of short date formats into a datetime.

    Supports:
      - "feb. 25" (assumes DEFAULT_YEAR)
      - "mar. 10"
      - "08-05-2026" (dd-mm-yyyy)
      - "10-05-2026"
      - "20-05-2026"
      - "15-05-2026"
      - "08/05/2026"
      - "2026-05-08"
    """
    s = str(s).strip()
    if not s:
        raise ValueError("Empty date string")

    if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", s):
        return datetime.strptime(s, "%d-%m-%Y")

    match = re.fullmatch(r"([A-Za-z]+\.?)\s+(\d{1,2})", s)
    if match:
        mon_raw = match.group(1).lower()
        day = int(match.group(2))
        if mon_raw not in MONTH_MAP:
            raise ValueError(f"Unknown month token: {mon_raw!r} in {s!r}")
        month = MONTH_MAP[mon_raw]
        return datetime(DEFAULT_YEAR, month, day)

    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    raise ValueError(f"Unrecognized date format: {s!r}")


def main() -> None:
    """Create and save a Gantt chart as gantt.png."""
    df = pd.read_csv(StringIO(TSV_DATA), sep="\t")

    df["Start_dt"] = df["Start"].apply(parse_mixed_date)
    df["End_dt"] = df["End"].apply(parse_mixed_date)

    df["Duration_days"] = (df["End_dt"] - df["Start_dt"]).dt.days
    df.loc[df["Duration_days"] <= 0, "Duration_days"] = 1

    df = df.sort_values("Start_dt", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    y_positions = range(len(df))
    starts = mdates.date2num(df["Start_dt"])
    durations = df["Duration_days"].to_numpy()

    ax.barh(y_positions, durations, left=starts)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["Task"])
    ax.invert_yaxis()

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.xticks(rotation=30, ha="right")

    ax.set_title("Project Gantt Chart (Feb–Jun 2026)")
    ax.set_xlabel("Date")

    plt.tight_layout()
    out = "gantt.png"
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

