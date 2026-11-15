#!/usr/bin/env python3
"""Aggregate per-run benchmark rows into one row per experiment."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


NUMERIC_COLUMNS = {
    "execution_time_s": "mean_execution_time_s",
    "speedup": "mean_speedup",
    "efficiency": "mean_efficiency_pct",
    "avg_rank_comp_time_s": "mean_avg_rank_comp_time_s",
    "avg_rank_comm_time_s": "mean_avg_rank_comm_time_s",
    "avg_rank_halo_exchange_time_s": "mean_avg_rank_halo_exchange_time_s",
    "avg_rank_halo_post_time_s": "mean_avg_rank_halo_post_time_s",
    "avg_rank_halo_wait_time_s": "mean_avg_rank_halo_wait_time_s",
    "avg_rank_allreduce_time_s": "mean_avg_rank_allreduce_time_s",
    # Optional fields retained when aggregating older CSVs.
    "max_rank_comp_time_s": "mean_max_rank_comp_time_s",
    "max_rank_comm_time_s": "mean_max_rank_comm_time_s",
}


def number(value):
    value = value.strip().rstrip("%")
    if not value or value.upper() in {"NA", "N/A", "NAN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="results/results/all_results.csv")
    parser.add_argument(
        "output",
        nargs="?",
        help="output CSV (default: all_results_aggregated.csv beside the input)",
    )
    parser.add_argument(
        "--exclude-rank2-outliers",
        action="store_true",
        help="for P=2 experiments, exclude runs taking more than 2x that experiment's median time",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name("all_results_aggregated.csv")
    )
    groups = defaultdict(list)

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "experiment", "source_file", "mpi_ranks", "run",
            "execution_time_s", "speedup", "efficiency",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
        numeric_columns = [
            (source, destination)
            for source, destination in NUMERIC_COLUMNS.items()
            if source in (reader.fieldnames or [])
        ]
        for row in reader:
            groups[row["experiment"]].append(row)

    output_columns = [
        "experiment", "source_file", "mpi_ranks", "runs", "excluded_runs", "excluded_run_ids",
        *(destination for _, destination in numeric_columns),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_columns)
        writer.writeheader()
        for experiment, rows in groups.items():
            first = rows[0]
            excluded = []
            included = rows
            if args.exclude_rank2_outliers and first["mpi_ranks"].strip() == "2":
                timed_rows = [
                    (row, number(row.get("execution_time_s", "")))
                    for row in rows
                ]
                times = [time for _, time in timed_rows if time is not None]
                if times:
                    cutoff = 2 * median(times)
                    excluded = [
                        row for row, time in timed_rows
                        if time is not None and time > cutoff
                    ]
                    excluded_ids = {row["run"] for row in excluded}
                    included = [row for row in rows if row["run"] not in excluded_ids]
            result = {
                "experiment": experiment,
                "source_file": first["source_file"],
                "mpi_ranks": first["mpi_ranks"],
                "runs": len(included),
                "excluded_runs": len(excluded),
                "excluded_run_ids": ";".join(row["run"] for row in excluded),
            }
            for source_column, output_column in numeric_columns:
                values = [number(row[source_column]) for row in included]
                values = [value for value in values if value is not None]
                result[output_column] = (
                    f"{sum(values) / len(values):.6f}" if values else "NA"
                )
            writer.writerow(result)

    print(f"Wrote {len(groups)} experiment summaries to {output_path}")


if __name__ == "__main__":
    main()
