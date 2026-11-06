#!/usr/bin/env python3
"""Aggregate per-run benchmark rows into one row per experiment."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


NUMERIC_COLUMNS = {
    "execution_time_s": "mean_execution_time_s",
    "speedup": "mean_speedup",
    "efficiency": "mean_efficiency_pct",
    "avg_rank_comp_time_s": "mean_avg_rank_comp_time_s",
    "max_rank_comp_time_s": "mean_max_rank_comp_time_s",
    "avg_rank_comm_time_s": "mean_avg_rank_comm_time_s",
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
    parser.add_argument("output", nargs="?", default="results/results/all_results_aggregated.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    groups = defaultdict(list)

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"experiment", "source_file", "mpi_ranks", "run", *NUMERIC_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            groups[row["experiment"]].append(row)

    output_columns = [
        "experiment", "source_file", "mpi_ranks", "runs",
        *NUMERIC_COLUMNS.values(),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_columns)
        writer.writeheader()
        for experiment, rows in groups.items():
            first = rows[0]
            result = {
                "experiment": experiment,
                "source_file": first["source_file"],
                "mpi_ranks": first["mpi_ranks"],
                "runs": len(rows),
            }
            for source_column, output_column in NUMERIC_COLUMNS.items():
                values = [number(row[source_column]) for row in rows]
                values = [value for value in values if value is not None]
                result[output_column] = (
                    f"{sum(values) / len(values):.6f}" if values else "NA"
                )
            writer.writerow(result)

    print(f"Wrote {len(groups)} experiment summaries to {output_path}")


if __name__ == "__main__":
    main()
