import json
import os
import time
import uuid
from pathlib import Path

from pymongo import MongoClient, ReadPreference
from pymongo.errors import PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern


URI = (
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,"
    "mongo4:27017,mongo5:27017/?replicaSet=rs0"
)

DB_NAME = "mw_partition_test_db"
COLL_NAME = "events"
NODES = ["mongo1", "mongo2", "mongo3", "mongo4", "mongo5"]

# Start with 1. Increase to 5, then 30 after the workflow is stable.
NUM_ITERATIONS = 5

COORD_DIR = Path("/app/mw_partition_coord")
SIGNAL_TIMEOUT_SECONDS = 240
ELECTION_TIMEOUT_SECONDS = 90
RECOVERY_TIMEOUT_SECONDS = 120


SCENARIOS = [
    {
        "id": "NP1",
        "name": "W1 before partition: R=majority, W=majority",
        "timing": "before",
        "read_concern": ReadConcern("majority"),
        "w1_write_concern": WriteConcern(w="majority", wtimeout=10000),
        "w2_write_concern": WriteConcern(w="majority", wtimeout=30000),
    },
    {
        "id": "NP2",
        "name": "W1 before partition: R=local, W=majority",
        "timing": "before",
        "read_concern": ReadConcern("local"),
        "w1_write_concern": WriteConcern(w="majority", wtimeout=10000),
        "w2_write_concern": WriteConcern(w="majority", wtimeout=30000),
    },
    {
        "id": "NP3",
        "name": "W1 before partition: R=local, W=1",
        "timing": "before",
        "read_concern": ReadConcern("local"),
        "w1_write_concern": WriteConcern(w=1),
        "w2_write_concern": WriteConcern(w=1),
    },
    {
        "id": "NP4",
        "name": "W1 after partition on old primary: R=local, W=1",
        "timing": "after",
        "read_concern": ReadConcern("local"),
        "w1_write_concern": WriteConcern(w=1),
        "w2_write_concern": WriteConcern(w=1),
    },
    {
        "id": "NP5",
        "name": "W1 after partition on old primary: R=local, W=majority",
        "timing": "after",
        "read_concern": ReadConcern("local"),
        # W1 is expected to fail quickly on the two-node minority side.
        "w1_write_concern": WriteConcern(w="majority", wtimeout=3000),
        # W2 is sent to the valid majority side and gets a normal deadline.
        "w2_write_concern": WriteConcern(w="majority", wtimeout=30000),
    },
]


def atomic_write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def signal_path(step, name):
    return COORD_DIR / f"step_{step:03d}_{name}.signal"


def create_signal(step, name):
    atomic_write(signal_path(step, name), "ready\n")


def wait_for_signal(step, name, timeout=SIGNAL_TIMEOUT_SECONDS):
    path = signal_path(step, name)
    started = time.time()
    while time.time() - started < timeout:
        if path.exists():
            return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {path.name}")


def prepare_coordination(total_steps):
    COORD_DIR.mkdir(parents=True, exist_ok=True)
    for path in COORD_DIR.iterdir():
        if path.is_file():
            path.unlink()
    atomic_write(COORD_DIR / "total_steps.txt", str(total_steps))


def get_primary(client):
    client.admin.command("ping")
    primary = client.primary
    if primary is None:
        raise RuntimeError("No primary is currently visible")
    return primary[0]


def read_majority_nodes(step):
    path = COORD_DIR / f"step_{step:03d}_majority.txt"
    return [node.strip() for node in path.read_text().splitlines() if node.strip()]


def wait_for_new_majority_primary(client, old_primary, majority_nodes):
    started = time.time()
    last_error = None
    while time.time() - started < ELECTION_TIMEOUT_SECONDS:
        try:
            primary = get_primary(client)
            if primary != old_primary and primary in majority_nodes:
                return primary
        except PyMongoError as error:
            last_error = error
        time.sleep(0.5)
    raise TimeoutError(
        "No new primary appeared on the three-node side. "
        f"Last error: {last_error}"
    )


def wait_for_majority_side_ready(client, majority_nodes, timeout_seconds=45):
    """Wait until the new primary reports all three partition members healthy."""
    started = time.time()
    last_status = None
    expected = set(majority_nodes)
    while time.time() - started < timeout_seconds:
        try:
            status = client.admin.command("replSetGetStatus")
            visible = {
                member.get("name", "").split(":")[0]
                for member in status.get("members", [])
                if member.get("health") == 1
                and member.get("stateStr") in ("PRIMARY", "SECONDARY")
            }
            last_status = visible
            if expected.issubset(visible):
                return
        except PyMongoError as error:
            last_status = str(error)
        time.sleep(0.5)
    raise TimeoutError(
        "The three-node majority side did not become replication-ready. "
        f"Last visible state: {last_status}"
    )


def wait_for_five_healthy_members(client):
    started = time.time()
    last_status = None
    while time.time() - started < RECOVERY_TIMEOUT_SECONDS:
        try:
            status = client.admin.command("replSetGetStatus")
            members = status.get("members", [])
            last_status = [
                (m.get("name"), m.get("stateStr"), m.get("health"))
                for m in members
            ]
            healthy = [
                m for m in members
                if m.get("health") == 1
                and m.get("stateStr") in ("PRIMARY", "SECONDARY")
            ]
            if len(healthy) == 5:
                return
        except PyMongoError as error:
            last_status = str(error)
        time.sleep(1)
    raise TimeoutError(f"Five-node recovery timed out: {last_status}")


def transfer_session_time(source_session, destination_session):
    if source_session.cluster_time is not None:
        destination_session.advance_cluster_time(source_session.cluster_time)
    if source_session.operation_time is not None:
        destination_session.advance_operation_time(source_session.operation_time)


def make_event(iteration_id, sequence):
    return {
        "_id": f"{iteration_id}-W{sequence}",
        "iteration_id": iteration_id,
        "client_id": "client_A",
        "sequence": sequence,
        "operation": f"W{sequence}",
        "must_follow": "W1" if sequence == 2 else None,
        "written_at": time.time(),
    }


def attempt_write(collection, document, session):
    started = time.perf_counter()
    try:
        result = collection.insert_one(document, session=session)
        return {
            "acknowledged": result.acknowledged,
            "error": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
    except PyMongoError as error:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "acknowledged": False,
            "error": f"{type(error).__name__}: {error}",
            "latency_ms": elapsed_ms,
        }


def run_iteration(client, scenario, step):
    iteration_id = str(uuid.uuid4())
    old_primary = get_primary(client)
    database = client[DB_NAME]
    cluster_write = database.get_collection(
        COLL_NAME, write_concern=scenario["w1_write_concern"]
    )
    read_collection = database.get_collection(
        COLL_NAME, read_concern=scenario["read_concern"]
    ).with_options(read_preference=ReadPreference.PRIMARY)

    atomic_write(COORD_DIR / f"step_{step:03d}_primary.txt", old_primary)
    atomic_write(COORD_DIR / f"step_{step:03d}_scenario.txt", scenario["id"])

    print("\n" + "-" * 78)
    print(f"Step {step}: {scenario['id']} — {scenario['name']}")
    print(f"Old primary: {old_primary}")

    w1 = {"acknowledged": False, "error": None, "latency_ms": 0.0}
    w2 = {"acknowledged": False, "error": None, "latency_ms": 0.0}
    partition_started = None

    with client.start_session(causal_consistency=True) as main_session:
        if scenario["timing"] == "before":
            w1 = attempt_write(
                cluster_write, make_event(iteration_id, 1), main_session
            )
            print(f"W1 before partition: {w1}")
            if not w1["acknowledged"]:
                raise RuntimeError(f"Pre-partition W1 unexpectedly failed: {w1['error']}")

        create_signal(step, "partition_ready")
        wait_for_signal(step, "partition_created")
        partition_started = time.perf_counter()

        majority_nodes = read_majority_nodes(step)
        print(f"Minority old primary: {old_primary}")
        print(f"Majority side: {majority_nodes}")

        if scenario["timing"] == "after":
            direct_uri = f"mongodb://{old_primary}:27017/?directConnection=true"
            direct_client = MongoClient(
                direct_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=15000,
                retryWrites=False,
            )
            direct_collection = direct_client[DB_NAME].get_collection(
                COLL_NAME, write_concern=scenario["w1_write_concern"]
            )
            try:
                with direct_client.start_session(causal_consistency=True) as old_session:
                    w1 = attempt_write(
                        direct_collection,
                        make_event(iteration_id, 1),
                        old_session,
                    )
                    # Only an acknowledged W1 establishes the dependency
                    # that W2 must follow. A timed-out majority write in NP5
                    # is classified as unavailable, not as a causal predecessor.
                    if w1["acknowledged"]:
                        transfer_session_time(old_session, main_session)
            finally:
                direct_client.close()
            print(f"W1 on old minority primary: {w1}")

        new_primary = wait_for_new_majority_primary(
            client, old_primary, majority_nodes
        )
        election_ms = (time.perf_counter() - partition_started) * 1000
        print(f"New majority-side primary: {new_primary}")

        # During a partition the original replica-set MongoClient can briefly
        # retain a confused view containing both the old and new primaries.
        # Route W2 directly to the new majority-side primary so the experiment
        # measures MongoDB's write concern instead of driver rediscovery timing.
        new_primary_uri = (
            f"mongodb://{new_primary}:27017/?directConnection=true"
        )
        new_primary_client = MongoClient(
            new_primary_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=15000,
            retryWrites=False,
        )
        new_primary_client.admin.command("ping")
        print("Waiting for all three majority-side members to be replication-ready...")
        wait_for_majority_side_ready(new_primary_client, majority_nodes)
        new_primary_collection = new_primary_client[DB_NAME].get_collection(
            COLL_NAME, write_concern=scenario["w2_write_concern"]
        )
        try:
            with new_primary_client.start_session(
                causal_consistency=True
            ) as new_primary_session:
                # Preserve the logical client's causal time across the two
                # physical MongoClient connections.
                transfer_session_time(main_session, new_primary_session)
                w2 = attempt_write(
                    new_primary_collection,
                    make_event(iteration_id, 2),
                    new_primary_session,
                )
        finally:
            new_primary_client.close()
        print(f"W2 on new primary: {w2}")

        create_signal(step, "heal_ready")
        wait_for_signal(step, "partition_healed")
        print("Partition healed. Waiting for five healthy members...")
        wait_for_five_healthy_members(client)

        # Allow rollback/synchronization to settle before classifying the history.
        time.sleep(2)

    # Verification is deliberately outside the causal operation session.
    # Otherwise a stale session timestamp left by a partitioned branch can
    # make this diagnostic read wait indefinitely. The tested operations W1
    # and W2 themselves still use the causally consistent session above.
    print("Five members healthy. Performing final verification read...")
    read_started = time.perf_counter()
    documents = list(
        read_collection.find({"iteration_id": iteration_id}).max_time_ms(10000)
    )
    read_ms = (time.perf_counter() - read_started) * 1000

    sequences = sorted(document["sequence"] for document in documents)
    w1_exists = 1 in sequences
    w2_exists = 2 in sequences
    both_acknowledged = w1["acknowledged"] and w2["acknowledged"]

    if both_acknowledged and w1_exists and w2_exists:
        classification = "PASS"
        valid_comparison = True
        violation = False
    elif both_acknowledged and not w1_exists and w2_exists:
        classification = "VIOLATION: W2 survived but W1 was rolled back"
        valid_comparison = True
        violation = True
    elif not w1["acknowledged"]:
        classification = "W1 UNAVAILABLE: no monotonic-write comparison"
        valid_comparison = False
        violation = False
    elif not w2["acknowledged"]:
        classification = "W2 UNAVAILABLE: no monotonic-write comparison"
        valid_comparison = False
        violation = False
    else:
        classification = "INDETERMINATE"
        valid_comparison = False
        violation = False

    print(f"Surviving sequences: {sequences}")
    print(f"Classification: {classification}")
    create_signal(step, "done")

    return {
        "scenario_id": scenario["id"],
        "scenario": scenario["name"],
        "w1_acknowledged": w1["acknowledged"],
        "w2_acknowledged": w2["acknowledged"],
        "w1_error": w1["error"],
        "w2_error": w2["error"],
        "w1_exists": w1_exists,
        "w2_exists": w2_exists,
        "valid_comparison": valid_comparison,
        "violation": violation,
        "classification": classification,
        "w1_ms": w1["latency_ms"],
        "w2_ms": w2["latency_ms"],
        "election_ms": election_ms,
        "read_ms": read_ms,
    }


def average(results, key):
    return sum(result[key] for result in results) / len(results) if results else 0.0


def summarize(scenario, results):
    valid = [result for result in results if result["valid_comparison"]]
    violations = sum(result["violation"] for result in valid)
    passes = len(valid) - violations
    acknowledged_operations = sum(
        int(result["w1_acknowledged"]) + int(result["w2_acknowledged"])
        for result in results
    )
    attempted_operations = 2 * len(results)
    return {
        "id": scenario["id"],
        "consistency": (passes / len(valid) * 100) if valid else None,
        "availability": (
            acknowledged_operations / attempted_operations * 100
            if attempted_operations else 0.0
        ),
        "violations": violations,
        "valid": len(valid),
        "avg_w1": average(results, "w1_ms"),
        "avg_w2": average(results, "w2_ms"),
        "avg_election": average(results, "election_ms"),
        "avg_read": average(results, "read_ms"),
    }


def print_summary(summaries):
    print("\n" + "=" * 126)
    print("MONOTONIC WRITES — NETWORK PARTITION SUMMARY")
    print("=" * 126)
    print(
        f"{'Case':<5} | {'MW Pass':<9} | {'Available':<10} | "
        f"{'Violations':<10} | {'Valid':<5} | {'Avg W1':<11} | "
        f"{'Avg W2':<11} | {'Election':<12} | {'Avg Read':<11}"
    )
    print("-" * 126)
    for item in summaries:
        consistency = "N/A" if item["consistency"] is None else f"{item['consistency']:.1f}%"
        print(
            f"{item['id']:<5} | {consistency:>9} | "
            f"{item['availability']:>8.1f}% | {item['violations']:>10} | "
            f"{item['valid']:>5} | {item['avg_w1']:>7.2f} ms | "
            f"{item['avg_w2']:>7.2f} ms | {item['avg_election']:>8.2f} ms | "
            f"{item['avg_read']:>7.2f} ms"
        )
    print("=" * 126)


def main():
    total_steps = len(SCENARIOS) * NUM_ITERATIONS
    prepare_coordination(total_steps)
    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=15000,
        retryWrites=False,
    )
    client.admin.command("ping")
    client.drop_database(DB_NAME)
    print(f"Connected. Initial primary: {get_primary(client)}")

    all_results = []
    summaries = []
    step = 0
    for scenario in SCENARIOS:
        scenario_results = []
        for _ in range(NUM_ITERATIONS):
            step += 1
            result = run_iteration(client, scenario, step)
            scenario_results.append(result)
            all_results.append(result)
        summaries.append(summarize(scenario, scenario_results))

    print_summary(summaries)
    atomic_write(
        COORD_DIR / "results.json",
        json.dumps(all_results, indent=2),
    )
    client.drop_database(DB_NAME)
    client.close()
    atomic_write(COORD_DIR / "experiment_complete.signal", "complete\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        COORD_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write(COORD_DIR / "fatal_error.txt", repr(error))
        raise
