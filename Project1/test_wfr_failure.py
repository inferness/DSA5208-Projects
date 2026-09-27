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
DB_NAME = "wfr_failure_test_db"
COLL_NAME = "events"
NUM_ITERATIONS = 20
COORD_DIR = Path("/app/wfr_failure_coord")
SIGNAL_TIMEOUT_SECONDS = 240
FAILOVER_TIMEOUT_SECONDS = 90
RECOVERY_TIMEOUT_SECONDS = 120
READ_VISIBILITY_TIMEOUT_SECONDS = 10


SCENARIOS = [
    {
        "id": "PF1",
        "name": "R=majority, W=majority",
        "read_concern": ReadConcern("majority"),
        "write_concern": WriteConcern(w="majority", wtimeout=10000),
    },
    {
        "id": "PF2",
        "name": "R=majority, W=1",
        "read_concern": ReadConcern("majority"),
        "write_concern": WriteConcern(w=1),
    },
    {
        "id": "PF3",
        "name": "R=local, W=majority",
        "read_concern": ReadConcern("local"),
        "write_concern": WriteConcern(w="majority", wtimeout=10000),
    },
    {
        "id": "PF4",
        "name": "R=local, W=1",
        "read_concern": ReadConcern("local"),
        "write_concern": WriteConcern(w=1),
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


def wait_for_new_primary(client, old_primary):
    started = time.time()
    last_error = None
    while time.time() - started < FAILOVER_TIMEOUT_SECONDS:
        try:
            primary = get_primary(client)
            if primary != old_primary:
                return primary
        except PyMongoError as error:
            last_error = error
        time.sleep(0.5)
    raise TimeoutError(f"No new primary was elected. Last error: {last_error}")


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
            primary_count = sum(
                member.get("stateStr") == "PRIMARY" for member in healthy
            )
            if len(healthy) == 5 and primary_count == 1:
                return
        except PyMongoError as error:
            last_status = str(error)
        time.sleep(1)
    raise TimeoutError(f"Five-node recovery timed out: {last_status}")


def verification_read_with_retry(collection, iteration_id, timeout_seconds=60):
    """Retry a diagnostic primary read during short post-restart elections."""
    started = time.time()
    last_error = None
    while time.time() - started < timeout_seconds:
        try:
            return list(collection.find({"iteration_id": iteration_id}))
        except PyMongoError as error:
            last_error = error
            time.sleep(1)
    raise TimeoutError(
        "Final verification could not reach a primary. "
        f"Last error: {last_error}"
    )


def timed(operation):
    started = time.perf_counter()
    try:
        value = operation()
        return value, None, (time.perf_counter() - started) * 1000
    except PyMongoError as error:
        return None, f"{type(error).__name__}: {error}", (
            time.perf_counter() - started
        ) * 1000


def read_until_visible(collection, document_id, session):
    """Retry the configured read until W1 is visible or the deadline expires."""
    started = time.perf_counter()
    deadline = time.monotonic() + READ_VISIBILITY_TIMEOUT_SECONDS
    last_error = None

    while time.monotonic() < deadline:
        try:
            document = collection.find_one(
                {"_id": document_id}, session=session, max_time_ms=2000
            )
            if document is not None:
                return document, None, (time.perf_counter() - started) * 1000
        except PyMongoError as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(0.1)

    return None, last_error, (time.perf_counter() - started) * 1000


def run_iteration(client, scenario, step, iteration_number):
    iteration_id = str(uuid.uuid4())
    old_primary = get_primary(client)
    database = client[DB_NAME]
    write_collection = database.get_collection(
        COLL_NAME, write_concern=scenario["write_concern"]
    )
    read_collection = database.get_collection(
        COLL_NAME, read_concern=scenario["read_concern"]
    ).with_options(read_preference=ReadPreference.PRIMARY)

    atomic_write(COORD_DIR / f"step_{step:03d}_primary.txt", old_primary)
    print("\n" + "-" * 76)
    print(f"Step {step}: {scenario['id']} — {scenario['name']}")
    print(f"Old primary: {old_primary}; iteration ID: {iteration_id}")

    w1_document = {
        "_id": f"{iteration_id}-W1",
        "iteration_id": iteration_id,
        "operation": "W1",
        "sequence": 1,
        "writer": "client_A",
        "value": 100,
        "written_at": time.time(),
    }

    with client.start_session(causal_consistency=True) as producer_session:
        w1_result, w1_error, w1_ms = timed(
            lambda: write_collection.insert_one(w1_document, session=producer_session)
        )
    w1_acknowledged = bool(w1_result and w1_result.acknowledged)

    observed_w1 = None
    read_error = None
    read_ms = 0.0
    w2_result = None
    w2_error = None
    w2_ms = 0.0
    new_primary = None

    # The same Client B session spans the read, primary failure, and W2.
    with client.start_session(causal_consistency=True) as client_b_session:
        if w1_acknowledged:
            observed_w1, read_error, read_ms = read_until_visible(
                read_collection,
                w1_document["_id"],
                client_b_session,
            )
        read_succeeded = observed_w1 is not None
        print(
            f"W1 acknowledged={w1_acknowledged}; "
            f"Client B read W1={read_succeeded}"
        )

        create_signal(step, "stop_ready")
        wait_for_signal(step, "primary_stopped")
        failover_started = time.perf_counter()
        new_primary = wait_for_new_primary(client, old_primary)
        election_ms = (time.perf_counter() - failover_started) * 1000
        print(f"New primary: {new_primary}")

        if read_succeeded:
            w2_document = {
                "_id": f"{iteration_id}-W2",
                "iteration_id": iteration_id,
                "operation": "W2",
                "sequence": 2,
                "writer": "client_B",
                "depends_on": observed_w1["_id"],
                "source_value_seen": observed_w1["value"],
                "written_at": time.time(),
            }
            w2_result, w2_error, w2_ms = timed(
                lambda: write_collection.insert_one(
                    w2_document, session=client_b_session
                )
            )

    w2_acknowledged = bool(w2_result and w2_result.acknowledged)
    create_signal(step, "restart_ready")
    wait_for_signal(step, "primary_restarted")
    wait_for_five_healthy_members(client)
    time.sleep(2)

    # Diagnostic verification is outside Client B's causal session.
    verify_collection = database.get_collection(
        COLL_NAME, read_concern=ReadConcern("majority")
    ).with_options(read_preference=ReadPreference.PRIMARY)
    documents, verify_error, verify_ms = timed(
        lambda: verification_read_with_retry(
            verify_collection, iteration_id
        )
    )
    documents = documents or []
    operations = {document.get("operation") for document in documents}
    w1_exists = "W1" in operations
    w2_exists = "W2" in operations

    if not w1_acknowledged:
        classification = "W1_UNAVAILABLE"
        valid = False
        violation = False
    elif not read_succeeded:
        classification = "READ_UNAVAILABLE"
        valid = False
        violation = False
    elif not w2_acknowledged:
        classification = "W2_UNAVAILABLE"
        valid = False
        violation = False
    elif w2_exists and not w1_exists:
        classification = "VIOLATION"
        valid = True
        violation = True
    elif w1_exists and w2_exists:
        classification = "PASS"
        valid = True
        violation = False
    else:
        classification = "INDETERMINATE"
        valid = False
        violation = False

    print(
        f"Result: {classification}; W1={w1_exists}, "
        f"read={read_succeeded}, W2={w2_exists}"
    )
    create_signal(step, "done")
    return {
        "scenario_id": scenario["id"],
        "scenario": scenario["name"],
        "iteration": iteration_number,
        "iteration_id": iteration_id,
        "old_primary": old_primary,
        "new_primary": new_primary,
        "w1_acknowledged": w1_acknowledged,
        "read_succeeded": read_succeeded,
        "w2_acknowledged": w2_acknowledged,
        "w1_exists": w1_exists,
        "w2_exists": w2_exists,
        "valid_comparison": valid,
        "violation": violation,
        "classification": classification,
        "w1_error": w1_error,
        "read_error": read_error,
        "w2_error": w2_error,
        "verify_error": verify_error,
        "w1_ms": w1_ms,
        "read_ms": read_ms,
        "w2_ms": w2_ms,
        "election_ms": election_ms,
        "verify_ms": verify_ms,
    }


def average(results, key):
    return sum(result[key] for result in results) / len(results) if results else 0.0


def summarize(scenario, results):
    valid = [result for result in results if result["valid_comparison"]]
    violations = sum(result["violation"] for result in valid)
    passes = len(valid) - violations
    successful_operations = sum(
        int(result["w1_acknowledged"])
        + int(result["read_succeeded"])
        + int(result["w2_acknowledged"])
        for result in results
    )
    return {
        "id": scenario["id"],
        "configuration": scenario["name"],
        "consistency": passes / len(valid) * 100 if valid else None,
        "availability": successful_operations / (3 * len(results)) * 100,
        "violations": violations,
        "valid": len(valid),
        "avg_w1": average(results, "w1_ms"),
        "avg_read": average(results, "read_ms"),
        "avg_w2": average(results, "w2_ms"),
        "avg_election": average(results, "election_ms"),
        "avg_verify": average(results, "verify_ms"),
    }


def print_summary(summaries):
    print("\n" + "=" * 178)
    print("WRITES-FOLLOW-READS — PRIMARY FAILURE SUMMARY")
    print("=" * 178)
    print(
        f"{'Case':<5} | {'Configuration':<25} | {'WFR Pass':<9} | {'Available':<10} | "
        f"{'Violations':<10} | {'Valid':<5} | {'Avg W1':<11} | "
        f"{'Avg Read':<11} | {'Avg W2':<11} | {'Election':<12} | {'Verify':<11}"
    )
    print("-" * 178)
    for item in summaries:
        consistency = "N/A" if item["consistency"] is None else f"{item['consistency']:.1f}%"
        print(
            f"{item['id']:<5} | {item['configuration']:<25} | {consistency:>9} | {item['availability']:>8.1f}% | "
            f"{item['violations']:>10} | {item['valid']:>5} | "
            f"{item['avg_w1']:>7.2f} ms | {item['avg_read']:>7.2f} ms | "
            f"{item['avg_w2']:>7.2f} ms | {item['avg_election']:>8.2f} ms | "
            f"{item['avg_verify']:>7.2f} ms"
        )
    print("=" * 178)


def main():
    total_steps = len(SCENARIOS) * NUM_ITERATIONS
    prepare_coordination(total_steps)
    client = MongoClient(
        URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=30000,
        retryWrites=False,
    )
    client.admin.command("ping")
    client.drop_database(DB_NAME)
    print(f"Connected. Initial primary: {get_primary(client)}")
    print(f"Total primary-failure steps: {total_steps}")

    all_results = []
    summaries = []
    step = 0
    for scenario in SCENARIOS:
        scenario_results = []
        for iteration in range(1, NUM_ITERATIONS + 1):
            step += 1
            result = run_iteration(client, scenario, step, iteration)
            scenario_results.append(result)
            all_results.append(result)
        summaries.append(summarize(scenario, scenario_results))

    print_summary(summaries)
    atomic_write(
        COORD_DIR / "results.json", json.dumps(all_results, indent=2)
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
