#!/usr/bin/env python3
"""Aggregate per-run weak-scaling results to one row per configuration."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


MEAN_FIELDS = (
    "execution_time_s",
    "weak_scaling_efficiency",
    "avg_rank_comp_time_s",
    "max_rank_comp_time_s",
    "avg_rank_comm_time_s",
    "max_rank_comm_time_s",
    "total_jacobi_iters",
    "mean_jacobi_iters_per_step",
    "global_cell_updates_per_s",
)


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, places=6):
    return f"{value:.{places}f}" if value is not None else "NA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="results/weak_scaling/all_weak_scaling_results.csv")
    parser.add_argument("output", nargs="?", default="results/weak_scaling/all_weak_scaling_aggregated.csv")
    args = parser.parse_args()

    groups = defaultdict(list)
    with Path(args.input).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"N", "steps", "mpi_ranks", "run", "max_residual", *MEAN_FIELDS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            key = (int(row["N"]), int(row["steps"]), int(row["mpi_ranks"]))
            groups[key].append(row)

    fields = [
        "N", "steps", "mpi_ranks", "runs", "local_rows_per_rank",
        "owned_cells_per_rank", "min_active_cells_per_rank", "max_active_cells_per_rank",
        "target_cells_per_rank", "target_error_pct",
        "mean_execution_time_s", "stddev_execution_time_s",
        "min_execution_time_s", "max_execution_time_s",
        "mean_weak_scaling_efficiency",
        "mean_avg_rank_comp_time_s", "mean_max_rank_comp_time_s",
        "mean_avg_rank_comm_time_s", "mean_max_rank_comm_time_s",
        "mean_total_jacobi_iters", "mean_jacobi_iters_per_step",
        "mean_global_cell_updates_per_s", "max_residual",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for (n, steps, p), rows in sorted(groups.items()):
            result = {
                "N": n,
                "steps": steps,
                "mpi_ranks": p,
                "runs": len(rows),
            }
            for field in (
                "local_rows_per_rank", "owned_cells_per_rank",
                "min_active_cells_per_rank", "max_active_cells_per_rank",
                "target_cells_per_rank", "target_error_pct",
            ):
                result[field] = rows[0][field]
            for source in MEAN_FIELDS:
                values = [numeric(row[source]) for row in rows]
                values = [value for value in values if value is not None]
                output_name = "mean_weak_scaling_efficiency" if source == "weak_scaling_efficiency" else f"mean_{source}"
                result[output_name] = fmt(sum(values) / len(values)) if values else "NA"

            times = [float(row["execution_time_s"]) for row in rows]
            avg_time = sum(times) / len(times)
            variance = sum((value - avg_time) ** 2 for value in times) / (len(times) - 1) if len(times) > 1 else 0.0
            result["mean_execution_time_s"] = fmt(avg_time)
            result["stddev_execution_time_s"] = fmt(math.sqrt(variance))
            result["min_execution_time_s"] = fmt(min(times))
            result["max_execution_time_s"] = fmt(max(times))
            residuals = [numeric(row["max_residual"]) for row in rows]
            residuals = [value for value in residuals if value is not None]
            result["max_residual"] = f"{max(residuals):.9e}" if residuals else "NA"
            writer.writerow(result)

    print(f"Wrote {len(groups)} configuration summaries to {output}")


if __name__ == "__main__":
    main()
