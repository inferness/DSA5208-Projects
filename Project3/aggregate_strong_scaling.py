#!/usr/bin/env python3
"""Aggregate per-run strong-scaling results to one row per configuration."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


MEASUREMENTS = {
    "execution_time_s": "mean_execution_time_s",
    "speedup": "mean_speedup",
    "efficiency": "mean_efficiency",
    "avg_rank_comp_time_s": "mean_avg_rank_comp_time_s",
    "max_rank_comp_time_s": "mean_max_rank_comp_time_s",
    "avg_rank_comm_time_s": "mean_avg_rank_comm_time_s",
    "max_rank_comm_time_s": "mean_max_rank_comm_time_s",
    "total_jacobi_iters": "mean_total_jacobi_iters",
}


def parse_number(value):
    value = value.strip().rstrip("%")
    if not value or value.upper() in {"NA", "N/A", "NAN"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value):
    return f"{value:.6f}" if value is not None else "NA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="results/strong_scaling/all_strong_scaling_results.csv",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="results/strong_scaling/all_strong_scaling_aggregated.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    groups = defaultdict(list)

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "N", "steps", "mpi_ranks", "run", "execution_time_s", "speedup",
            "efficiency", "avg_rank_comp_time_s", "max_rank_comp_time_s",
            "avg_rank_comm_time_s", "max_rank_comm_time_s",
            "total_jacobi_iters", "max_residual",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            key = (int(row["N"]), int(row["steps"]), int(row["mpi_ranks"]))
            groups[key].append(row)

    columns = [
        "N", "steps", "mpi_ranks", "runs",
        "mean_execution_time_s", "stddev_execution_time_s",
        "min_execution_time_s", "max_execution_time_s",
        "mean_speedup", "mean_efficiency",
        "mean_avg_rank_comp_time_s", "mean_max_rank_comp_time_s",
        "mean_avg_rank_comm_time_s", "mean_max_rank_comm_time_s",
        "mean_total_jacobi_iters", "max_residual",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for (n, steps, ranks), rows in sorted(groups.items()):
            result = {"N": n, "steps": steps, "mpi_ranks": ranks, "runs": len(rows)}
            for source, destination in MEASUREMENTS.items():
                values = [parse_number(row[source]) for row in rows]
                values = [value for value in values if value is not None]
                result[destination] = fmt(mean(values))

            times = [float(row["execution_time_s"]) for row in rows]
            result["min_execution_time_s"] = fmt(min(times))
            result["max_execution_time_s"] = fmt(max(times))
            result["stddev_execution_time_s"] = fmt(
                math.sqrt(sum((time - mean(times)) ** 2 for time in times) / (len(times) - 1))
                if len(times) > 1 else 0.0
            )
            residuals = [parse_number(row["max_residual"]) for row in rows]
            residuals = [value for value in residuals if value is not None]
            result["max_residual"] = f"{max(residuals):.9e}" if residuals else "NA"
            writer.writerow(result)

    print(f"Wrote {len(groups)} configuration summaries to {output_path}")


if __name__ == "__main__":
    main()
