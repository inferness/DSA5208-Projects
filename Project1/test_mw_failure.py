import os
import time
import uuid
from pathlib import Path

from pymongo import MongoClient, ReadPreference
from pymongo.errors import PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern


# ------------------------------------------------------------
# MongoDB configuration
# ------------------------------------------------------------

URI = (
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,"
    "mongo4:27017,mongo5:27017/"
    "?replicaSet=rs0"
)

DB_NAME = "mw_failure_test_db"
COLL_NAME = "events"

# Start with 1 because every iteration stops and restarts a primary.
# Increase this after confirming that the experiment works.
NUM_ITERATIONS = 30

SERVER_SELECTION_TIMEOUT_MS = 5000
FAILOVER_TIMEOUT_SECONDS = 90
RECOVERY_TIMEOUT_SECONDS = 120
SIGNAL_TIMEOUT_SECONDS = 180

COORD_DIR = Path("/app/mw_coord")


# ------------------------------------------------------------
# Coordination-file helpers
# ------------------------------------------------------------

def signal_path(step_number, signal_name):
    return COORD_DIR / (
        f"step_{step_number:03d}_{signal_name}.signal"
    )


def write_text_file(path, content):
    """
    Write a file atomically so PowerShell does not read a
    partially written file.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


def create_signal(step_number, signal_name):
    path = signal_path(step_number, signal_name)
    write_text_file(path, "ready\n")


def wait_for_signal(
    step_number,
    signal_name,
    timeout_seconds=SIGNAL_TIMEOUT_SECONDS
):
    path = signal_path(step_number, signal_name)
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        if path.exists():
            return

        time.sleep(0.25)

    raise TimeoutError(
        f"Timed out waiting for signal: {path.name}"
    )


def prepare_coordination_directory(total_steps):
    COORD_DIR.mkdir(parents=True, exist_ok=True)

    # Remove only files produced by a previous experiment.
    for existing_file in COORD_DIR.glob("step_*"):
        if existing_file.is_file():
            existing_file.unlink()

    total_steps_file = COORD_DIR / "total_steps.txt"

    if total_steps_file.exists():
        total_steps_file.unlink()

    fatal_error_file = COORD_DIR / "fatal_error.txt"

    if fatal_error_file.exists():
        fatal_error_file.unlink()

    write_text_file(
        total_steps_file,
        str(total_steps)
    )


# ------------------------------------------------------------
# MongoDB helpers
# ------------------------------------------------------------

def get_primary_name(client):
    """
    Return a Docker container name such as 'mongo3'
    from PyMongo's ('mongo3', 27017) primary tuple.
    """
    client.admin.command("ping")

    primary = client.primary

    if primary is None:
        raise RuntimeError(
            "MongoDB currently has no detectable primary."
        )

    return primary[0]


def wait_for_new_primary(
    client,
    old_primary_name,
    timeout_seconds=FAILOVER_TIMEOUT_SECONDS
):
    print(
        f"  Waiting for a new primary after "
        f"{old_primary_name} stopped..."
    )

    start_time = time.time()
    last_error = None

    while time.time() - start_time < timeout_seconds:
        try:
            client.admin.command("ping")
            new_primary_name = get_primary_name(client)

            if new_primary_name != old_primary_name:
                print(
                    f"  New primary elected: "
                    f"{new_primary_name}"
                )
                return new_primary_name

        except PyMongoError as error:
            last_error = error

        time.sleep(1)

    raise TimeoutError(
        "No new primary was elected within "
        f"{timeout_seconds} seconds. "
        f"Last MongoDB error: {last_error}"
    )


def wait_for_five_healthy_members(
    client,
    timeout_seconds=RECOVERY_TIMEOUT_SECONDS
):
    print(
        "  Waiting for all five replica-set members "
        "to become healthy..."
    )

    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout_seconds:
        try:
            status = client.admin.command(
                "replSetGetStatus"
            )

            members = status.get("members", [])
            healthy_members = [
                member
                for member in members
                if member.get("health") == 1
                and member.get("stateStr")
                in ("PRIMARY", "SECONDARY")
            ]

            last_status = [
                (
                    member.get("name"),
                    member.get("stateStr"),
                    member.get("health")
                )
                for member in members
            ]

            if len(healthy_members) == 5:
                print(
                    "  All five members are healthy."
                )
                return

        except PyMongoError as error:
            last_status = str(error)

        time.sleep(1)

    raise TimeoutError(
        "The replica set did not return to five healthy "
        f"members. Last status: {last_status}"
    )


def cleanup_database(client):
    client.drop_database(DB_NAME)


# ------------------------------------------------------------
# One primary-failure iteration
# ------------------------------------------------------------

def run_failure_iteration(
    client,
    scenario_name,
    write_concern,
    read_concern,
    step_number
):
    """
    Run one monotonic-writes experiment:

    1. W1 is sent to the current primary.
    2. Python tells PowerShell that W1 was acknowledged.
    3. PowerShell stops the current primary.
    4. MongoDB elects a new primary.
    5. W2 is sent using the same client session.
    6. PowerShell restarts the old primary.
    7. Python verifies whether W1 and W2 survived.
    """
    iteration_id = str(uuid.uuid4())

    w1_id = f"{iteration_id}-W1"
    w2_id = f"{iteration_id}-W2"

    database = client[DB_NAME]

    write_collection = database.get_collection(
        COLL_NAME,
        write_concern=write_concern
    )

    read_collection = database.get_collection(
        COLL_NAME,
        read_concern=read_concern
    ).with_options(
        read_preference=ReadPreference.PRIMARY
    )

    old_primary_name = get_primary_name(client)

    print()
    print("-" * 72)
    print(f"Step:             {step_number}")
    print(f"Scenario:         {scenario_name}")
    print(f"Old primary:      {old_primary_name}")
    print(f"Iteration ID:     {iteration_id}")

    primary_file = (
        COORD_DIR
        / f"step_{step_number:03d}_primary.txt"
    )

    write_text_file(
        primary_file,
        old_primary_name
    )

    with client.start_session(
        causal_consistency=True
    ) as session:

        # ----------------------------------------------------
        # W1: first write
        # ----------------------------------------------------

        w1_start = time.perf_counter()

        w1_result = write_collection.insert_one(
            {
                "_id": w1_id,
                "iteration_id": iteration_id,
                "client_id": "client_A",
                "sequence": 1,
                "operation": "W1",
                "written_at": time.time()
            },
            session=session
        )

        w1_latency_ms = (
            time.perf_counter() - w1_start
        ) * 1000

        print(
            f"  W1 acknowledged by {old_primary_name} "
            f"in {w1_latency_ms:.3f} ms."
        )

        # Tell PowerShell that it should stop the primary.
        create_signal(
            step_number,
            "stop_ready"
        )

        print(
            f"  Waiting for PowerShell to stop "
            f"{old_primary_name}..."
        )

        wait_for_signal(
            step_number,
            "primary_stopped"
        )

        print(
            f"  PowerShell stopped {old_primary_name}."
        )

        # Wait for another replica-set member to become primary.
        new_primary_name = wait_for_new_primary(
            client,
            old_primary_name
        )

        # ----------------------------------------------------
        # W2: second write, using the same client session
        # ----------------------------------------------------

        w2_start = time.perf_counter()

        w2_result = write_collection.insert_one(
            {
                "_id": w2_id,
                "iteration_id": iteration_id,
                "client_id": "client_A",
                "sequence": 2,
                "operation": "W2",
                "must_follow": "W1",
                "written_at": time.time()
            },
            session=session
        )

        w2_latency_ms = (
            time.perf_counter() - w2_start
        ) * 1000

        print(
            f"  W2 acknowledged by {new_primary_name} "
            f"in {w2_latency_ms:.3f} ms."
        )

        # Tell PowerShell that it can restart the old primary.
        create_signal(
            step_number,
            "restart_ready"
        )

        print(
            f"  Waiting for PowerShell to restart "
            f"{old_primary_name}..."
        )

        wait_for_signal(
            step_number,
            "primary_restarted"
        )

        wait_for_five_healthy_members(client)

        # ----------------------------------------------------
        # Verification read from the current primary
        # ----------------------------------------------------

        read_start = time.perf_counter()

        surviving_documents = list(
            read_collection.find(
                {"iteration_id": iteration_id},
                session=session
            )
        )

        read_latency_ms = (
            time.perf_counter() - read_start
        ) * 1000

    surviving_sequences = sorted(
        document["sequence"]
        for document in surviving_documents
    )

    w1_exists = 1 in surviving_sequences
    w2_exists = 2 in surviving_sequences

    if w1_exists and w2_exists:
        outcome = "PASS"
        valid_comparison = True
        violation = False

    elif not w1_exists and w2_exists:
        outcome = "VIOLATION: W2 survived but W1 disappeared"
        valid_comparison = True
        violation = True

    elif w1_exists and not w2_exists:
        outcome = "W2 unavailable"
        valid_comparison = False
        violation = False

    else:
        outcome = "Both writes unavailable"
        valid_comparison = False
        violation = False

    print(f"  W1 survived:     {w1_exists}")
    print(f"  W2 survived:     {w2_exists}")
    print(f"  Final sequences: {surviving_sequences}")
    print(f"  Result:          {outcome}")

    create_signal(
        step_number,
        "done"
    )

    return {
        "scenario": scenario_name,
        "step_number": step_number,
        "old_primary": old_primary_name,
        "new_primary": new_primary_name,
        "w1_acknowledged": w1_result.acknowledged,
        "w2_acknowledged": w2_result.acknowledged,
        "w1_exists": w1_exists,
        "w2_exists": w2_exists,
        "valid_comparison": valid_comparison,
        "violation": violation,
        "outcome": outcome,
        "w1_latency_ms": w1_latency_ms,
        "w2_latency_ms": w2_latency_ms,
        "read_latency_ms": read_latency_ms
    }


# ------------------------------------------------------------
# Scenario calculations
# ------------------------------------------------------------

def summarize_scenario(scenario_name, results):
    valid_results = [
        result
        for result in results
        if result["valid_comparison"]
    ]

    violations = sum(
        1
        for result in valid_results
        if result["violation"]
    )

    passes = len(valid_results) - violations

    consistency_rate = (
        passes / len(valid_results) * 100
        if valid_results
        else 0.0
    )

    available_results = [
        result
        for result in results
        if result["w1_acknowledged"]
        and result["w2_acknowledged"]
    ]

    availability_rate = (
        len(available_results) / len(results) * 100
        if results
        else 0.0
    )

    avg_w1 = (
        sum(
            result["w1_latency_ms"]
            for result in results
        ) / len(results)
        if results
        else 0.0
    )

    avg_w2 = (
        sum(
            result["w2_latency_ms"]
            for result in results
        ) / len(results)
        if results
        else 0.0
    )

    avg_read = (
        sum(
            result["read_latency_ms"]
            for result in results
        ) / len(results)
        if results
        else 0.0
    )

    return {
        "scenario": scenario_name,
        "consistency_rate": consistency_rate,
        "availability_rate": availability_rate,
        "violations": violations,
        "valid_comparisons": len(valid_results),
        "avg_w1": avg_w1,
        "avg_w2": avg_w2,
        "avg_read": avg_read
    }


def print_summary_table(summaries):
    print()
    print("=" * 128)
    print(
        "                         "
        "MONOTONIC WRITES — PRIMARY FAILURE SUMMARY"
    )
    print("=" * 119)

    print(
        f"{'Case':<5} | "
        f"{'Configuration':<25} | "
        f"{'MW Pass':<9} | "
        f"{'Available':<10} | "
        f"{'Violations':<10} | "
        f"{'Avg W1':<11} | "
        f"{'Avg W2':<11} | "
        f"{'Avg Read':<11}"
    )

    print("-" * 128)

    for summary in summaries:
        print(
            f"{summary['id']:<5} | "
            f"{summary['configuration']:<25} | "
            f"{summary['consistency_rate']:>6.1f}%   | "
            f"{summary['availability_rate']:>6.1f}%    | "
            f"{summary['violations']:>10} | "
            f"{summary['avg_w1']:>7.3f} ms | "
            f"{summary['avg_w2']:>7.3f} ms | "
            f"{summary['avg_read']:>7.3f} ms"
        )

    print("=" * 128)


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

def main():
    scenarios = [
        {
            "id": "PF1",
            "name": "R=majority, W=majority",
            "read_concern": ReadConcern("majority"),
            "write_concern": WriteConcern(
                w="majority",
                wtimeout=10000
            )
        },
        {
            "id": "PF2",
            "name": "R=local, W=majority",
            "read_concern": ReadConcern("local"),
            "write_concern": WriteConcern(
                w="majority",
                wtimeout=10000
            )
        },
        {
            "id": "PF3",
            "name": "R=local, W=1",
            "read_concern": ReadConcern("local"),
            "write_concern": WriteConcern(w=1)
        }
    ]

    total_steps = len(scenarios) * NUM_ITERATIONS
    prepare_coordination_directory(total_steps)

    print("Connecting to the five-node replica set...")

    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        retryWrites=True
    )

    client.admin.command("ping")

    print("Connected successfully.")
    print(f"Initial primary: {get_primary_name(client)}")
    print(f"Total failure steps: {total_steps}")

    cleanup_database(client)

    step_number = 0
    all_summaries = []

    for scenario in scenarios:
        scenario_results = []

        print()
        print("=" * 72)
        print(f"Starting scenario: {scenario['name']}")
        print("=" * 72)

        for iteration in range(NUM_ITERATIONS):
            step_number += 1

            print(
                f"\nScenario iteration "
                f"{iteration + 1}/{NUM_ITERATIONS}"
            )

            result = run_failure_iteration(
                client=client,
                scenario_name=scenario["name"],
                write_concern=scenario["write_concern"],
                read_concern=scenario["read_concern"],
                step_number=step_number
            )

            scenario_results.append(result)

        summary = summarize_scenario(
            scenario["name"],
            scenario_results
        )

        summary["id"] = scenario["id"]
        summary["configuration"] = scenario["name"]

        all_summaries.append(summary)

    print_summary_table(all_summaries)

    cleanup_database(client)
    client.close()

    write_text_file(
        COORD_DIR / "experiment_complete.signal",
        "complete\n"
    )

    print()
    print("Experiment completed successfully.")
    print("Test database cleaned up.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        COORD_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        write_text_file(
            COORD_DIR / "fatal_error.txt",
            repr(error)
        )

        print()
        print("=" * 72)
        print("FATAL ERROR")
        print("=" * 72)
        print(repr(error))

        raise
