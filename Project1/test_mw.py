import time
import uuid

from pymongo import MongoClient, ReadPreference
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern


# Five-node MongoDB replica set inside the Docker network
URI = (
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,"
    "mongo4:27017,mongo5:27017/"
    "?replicaSet=rs0"
)

DB_NAME = "mw_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 50


def cleanup_db(client):
    """Remove the test database so each run starts clean."""
    client.drop_database(DB_NAME)


def run_iteration(client, write_concern, read_concern):
    """
    Run one monotonic-writes test:

    1. W1 creates a document with version 1.
    2. W2 changes the same document to version 2.
    3. Read the final document from the primary.
    4. Pass if W1 succeeded, W2 succeeded, and the final version is 2.

    All operations use the same causally consistent session.
    """
    db = client[DB_NAME]

    coll_write = db.get_collection(
        COLL_NAME,
        write_concern=write_concern
    )

    coll_read = db.get_collection(
        COLL_NAME,
        read_concern=read_concern
    ).with_options(
        read_preference=ReadPreference.PRIMARY
    )

    doc_id = str(uuid.uuid4())

    with client.start_session(causal_consistency=True) as session:

        # ---------------------------------------------------------
        # W1: Create the document with version 1
        # ---------------------------------------------------------
        w1_start = time.perf_counter()

        w1_result = coll_write.insert_one(
            {
                "_id": doc_id,
                "client_id": "client_A",
                "version": 1,
                "w1_time": time.time()
            },
            session=session
        )

        w1_latency_ms = (
            time.perf_counter() - w1_start
        ) * 1000

        # ---------------------------------------------------------
        # W2: Update the same document to version 2
        # ---------------------------------------------------------
        w2_start = time.perf_counter()

        w2_result = coll_write.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "version": 2,
                    "w2_time": time.time()
                }
            },
            session=session
        )

        w2_latency_ms = (
            time.perf_counter() - w2_start
        ) * 1000

        # ---------------------------------------------------------
        # Read the final state from the current primary
        # ---------------------------------------------------------
        read_start = time.perf_counter()

        final_document = coll_read.find_one(
            {"_id": doc_id},
            session=session
        )

        read_latency_ms = (
            time.perf_counter() - read_start
        ) * 1000

    observed_version = (
        final_document.get("version")
        if final_document is not None
        else None
    )

    # Monotonic writes are satisfied if W1 was acknowledged,
    # W2 found and updated the document, and the final version is 2.
    monotonic_writes_satisfied = (
        w1_result.acknowledged
        and w2_result.acknowledged
        and w2_result.matched_count == 1
        and observed_version == 2
    )

    return {
        "satisfied": monotonic_writes_satisfied,
        "observed_version": observed_version,
        "w1_latency_ms": w1_latency_ms,
        "w2_latency_ms": w2_latency_ms,
        "read_latency_ms": read_latency_ms
    }


def test_scenario(
    client,
    name,
    write_concern,
    read_concern,
    iterations=NUM_ITERATIONS
):
    """Run one configuration repeatedly and calculate its metrics."""

    print(
        f"\nRunning: [{name}] "
        f"({iterations} repetitions)..."
    )

    # Start every scenario with an empty collection
    client[DB_NAME][COLL_NAME].drop()

    passes = 0
    completed_iterations = 0
    failed_iterations = 0

    w1_latencies = []
    w2_latencies = []
    read_latencies = []

    scenario_start = time.perf_counter()

    for iteration in range(iterations):
        try:
            result = run_iteration(
                client=client,
                write_concern=write_concern,
                read_concern=read_concern
            )

            completed_iterations += 1

            w1_latencies.append(
                result["w1_latency_ms"]
            )

            w2_latencies.append(
                result["w2_latency_ms"]
            )

            read_latencies.append(
                result["read_latency_ms"]
            )

            if result["satisfied"]:
                passes += 1
            else:
                print(
                    f"\n  Iteration {iteration + 1}: "
                    f"monotonic-writes violation; "
                    f"observed version = "
                    f"{result['observed_version']}"
                )

        except Exception as error:
            failed_iterations += 1

            print(
                f"\n  Iteration {iteration + 1}: "
                f"operation unavailable: {error}"
            )

        print(
            f"\r  Progress: {iteration + 1}/{iterations}",
            end="",
            flush=True
        )

    total_duration_seconds = (
        time.perf_counter() - scenario_start
    )

    # Consistency is calculated only among completed iterations
    consistency_rate = (
        passes / completed_iterations * 100
        if completed_iterations > 0
        else 0.0
    )

    # Availability measures how many complete iterations succeeded
    availability_rate = (
        completed_iterations / iterations * 100
    )

    avg_w1_latency = (
        sum(w1_latencies) / len(w1_latencies)
        if w1_latencies
        else 0.0
    )

    avg_w2_latency = (
        sum(w2_latencies) / len(w2_latencies)
        if w2_latencies
        else 0.0
    )

    avg_read_latency = (
        sum(read_latencies) / len(read_latencies)
        if read_latencies
        else 0.0
    )

    print()
    print(
        f"  MW Pass Rate:       "
        f"{consistency_rate:6.1f}% "
        f"({passes}/{completed_iterations})"
    )
    print(
        f"  Availability:       "
        f"{availability_rate:6.1f}% "
        f"({completed_iterations}/{iterations})"
    )
    print(
        f"  Failed Iterations:  "
        f"{failed_iterations}"
    )
    print(
        f"  Avg W1 Latency:     "
        f"{avg_w1_latency:7.3f} ms"
    )
    print(
        f"  Avg W2 Latency:     "
        f"{avg_w2_latency:7.3f} ms"
    )
    print(
        f"  Avg Read Latency:   "
        f"{avg_read_latency:7.3f} ms"
    )
    print(
        f"  Total Duration:     "
        f"{total_duration_seconds:7.2f} s"
    )

    return {
        "scenario": name,
        "consistency_rate": consistency_rate,
        "availability_rate": availability_rate,
        "passes": passes,
        "completed": completed_iterations,
        "failed": failed_iterations,
        "avg_w1": avg_w1_latency,
        "avg_w2": avg_w2_latency,
        "avg_read": avg_read_latency
    }


def print_summary_table(results):
    """Print the aggregated results for every configuration."""

    print("\n" + "=" * 110)
    print(
        "                         "
        "MONOTONIC WRITES BENCHMARK SUMMARY"
    )
    print("=" * 110)

    print(
        f"{'Scenario':<34} | "
        f"{'MW Pass':<9} | "
        f"{'Available':<10} | "
        f"{'Avg W1':<11} | "
        f"{'Avg W2':<11} | "
        f"{'Avg Read':<11}"
    )

    print("-" * 110)

    for result in results:
        print(
            f"{result['scenario']:<34} | "
            f"{result['consistency_rate']:>6.1f}%   | "
            f"{result['availability_rate']:>6.1f}%    | "
            f"{result['avg_w1']:>7.3f} ms | "
            f"{result['avg_w2']:>7.3f} ms | "
            f"{result['avg_read']:>7.3f} ms"
        )

    print("=" * 110)


def main():
    print("Connecting to MongoDB five-node replica set...")

    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=5000
    )

    # Verify that the replica set is reachable
    client.admin.command("ping")

    primary = client.primary

    print("Connected successfully!")
    print(f"Current primary: {primary}")

    cleanup_db(client)

    test_cases = [
        {
            "name": "R=majority, W=majority",
            "read_concern": ReadConcern("majority"),
            "write_concern": WriteConcern(
                w="majority",
                wtimeout=5000
            )
        },
        {
            "name": "R=local, W=majority",
            "read_concern": ReadConcern("local"),
            "write_concern": WriteConcern(
                w="majority",
                wtimeout=5000
            )
        },
        {
            "name": "R=local, W=1",
            "read_concern": ReadConcern("local"),
            "write_concern": WriteConcern(w=1)
        }
    ]

    results = []

    for case in test_cases:
        result = test_scenario(
            client=client,
            name=case["name"],
            write_concern=case["write_concern"],
            read_concern=case["read_concern"]
        )

        results.append(result)

    print_summary_table(results)

    cleanup_db(client)
    print("\nTest database cleaned up successfully.")

    client.close()


if __name__ == "__main__":
    main()