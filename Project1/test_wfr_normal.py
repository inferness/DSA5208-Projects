import json
import time
import uuid

from pymongo import MongoClient, ReadPreference
from pymongo.errors import PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern


URI = (
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,"
    "mongo4:27017,mongo5:27017/?replicaSet=rs0"
)
DB_NAME = "wfr_normal_test_db"
COLL_NAME = "events"
NUM_ITERATIONS = 50
RESULTS_FILE = "wfr_normal_results.json"
VERIFICATION_TIMEOUT_SECONDS = 10


SCENARIOS = [
    {
        "id": "N1",
        "name": "R=majority, W=majority",
        "read_concern": ReadConcern("majority"),
        "write_concern": WriteConcern(w="majority", wtimeout=10000),
    },
    {
        "id": "N2",
        "name": "R=local, W=majority",
        "read_concern": ReadConcern("local"),
        "write_concern": WriteConcern(w="majority", wtimeout=10000),
    },
    {
        "id": "N3",
        "name": "R=local, W=1",
        "read_concern": ReadConcern("local"),
        "write_concern": WriteConcern(w=1),
    },
]


def timed(operation):
    started = time.perf_counter()
    try:
        value = operation()
        return value, None, (time.perf_counter() - started) * 1000
    except PyMongoError as error:
        return None, f"{type(error).__name__}: {error}", (
            time.perf_counter() - started
        ) * 1000


def wait_for_majority_visibility(collection, iteration_id):
    """
    Wait briefly for w:1 writes to reach the majority-committed snapshot.

    A majority read is not automatically causally linked to Client B's
    earlier w:1 write when this diagnostic read uses a fresh session. Without
    polling, it can legally return the older majority-committed snapshot and
    make an otherwise successful N3 iteration look indeterminate.
    """
    started = time.time()
    latest_documents = []

    while time.time() - started < VERIFICATION_TIMEOUT_SECONDS:
        latest_documents = list(
            collection.find({"iteration_id": iteration_id}).max_time_ms(5000)
        )
        operations = {
            document.get("operation") for document in latest_documents
        }
        if {"W1", "W2"}.issubset(operations):
            return latest_documents
        time.sleep(0.1)

    return latest_documents


def run_iteration(client, scenario, iteration_number):
    iteration_id = str(uuid.uuid4())
    database = client[DB_NAME]
    write_collection = database.get_collection(
        COLL_NAME, write_concern=scenario["write_concern"]
    )
    read_collection = database.get_collection(
        COLL_NAME, read_concern=scenario["read_concern"]
    ).with_options(read_preference=ReadPreference.PRIMARY)

    w1_document = {
        "_id": f"{iteration_id}-W1",
        "iteration_id": iteration_id,
        "operation": "W1",
        "sequence": 1,
        "writer": "client_A",
        "value": 100,
        "written_at": time.time(),
    }

    # Client A produces W1 in its own logical session.
    with client.start_session(causal_consistency=True) as producer_session:
        w1_result, w1_error, w1_ms = timed(
            lambda: write_collection.insert_one(
                w1_document, session=producer_session
            )
        )
    w1_acknowledged = bool(w1_result and w1_result.acknowledged)

    read_succeeded = False
    observed_w1 = None
    w2_result = None
    w2_error = None
    read_error = None
    read_ms = 0.0
    w2_ms = 0.0

    # Client B's read and dependent W2 use the same causal session.
    if w1_acknowledged:
        with client.start_session(causal_consistency=True) as client_b_session:
            observed_w1, read_error, read_ms = timed(
                lambda: read_collection.find_one(
                    {"_id": w1_document["_id"]}, session=client_b_session
                )
            )
            read_succeeded = observed_w1 is not None

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

    verify_collection = database.get_collection(
        COLL_NAME,
        read_concern=ReadConcern("majority"),
    ).with_options(read_preference=ReadPreference.PRIMARY)
    documents, verify_error, verify_ms = timed(
        lambda: wait_for_majority_visibility(
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
        f"  {scenario['id']} iteration {iteration_number}: {classification} "
        f"(W1={w1_exists}, read={read_succeeded}, W2={w2_exists})"
    )

    return {
        "scenario_id": scenario["id"],
        "scenario": scenario["name"],
        "iteration": iteration_number,
        "iteration_id": iteration_id,
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
        "consistency": passes / len(valid) * 100 if valid else None,
        "availability": successful_operations / (3 * len(results)) * 100,
        "violations": violations,
        "valid": len(valid),
        "avg_w1": average(results, "w1_ms"),
        "avg_read": average(results, "read_ms"),
        "avg_w2": average(results, "w2_ms"),
        "avg_verify": average(results, "verify_ms"),
    }


def print_summary(summaries):
    print("\n" + "=" * 128)
    print("WRITES-FOLLOW-READS — NORMAL OPERATION SUMMARY")
    print("=" * 128)
    print(
        f"{'Case':<5} | {'WFR Pass':<9} | {'Available':<10} | "
        f"{'Violations':<10} | {'Valid':<5} | {'Avg W1':<11} | "
        f"{'Avg Read':<11} | {'Avg W2':<11} | {'Verify':<11}"
    )
    print("-" * 128)
    for item in summaries:
        consistency = "N/A" if item["consistency"] is None else f"{item['consistency']:.1f}%"
        print(
            f"{item['id']:<5} | {consistency:>9} | {item['availability']:>8.1f}% | "
            f"{item['violations']:>10} | {item['valid']:>5} | "
            f"{item['avg_w1']:>7.2f} ms | {item['avg_read']:>7.2f} ms | "
            f"{item['avg_w2']:>7.2f} ms | {item['avg_verify']:>7.2f} ms"
        )
    print("=" * 128)


def main():
    client = MongoClient(URI, serverSelectionTimeoutMS=5000, retryWrites=True)
    client.admin.command("ping")
    client.drop_database(DB_NAME)
    print(f"Connected. Current primary: {client.primary}")

    all_results = []
    summaries = []
    for scenario in SCENARIOS:
        print(f"\nRunning {scenario['id']}: {scenario['name']}")
        scenario_results = [
            run_iteration(client, scenario, iteration)
            for iteration in range(1, NUM_ITERATIONS + 1)
        ]
        all_results.extend(scenario_results)
        summaries.append(summarize(scenario, scenario_results))

    print_summary(summaries)
    with open(RESULTS_FILE, "w", encoding="utf-8") as output:
        json.dump(all_results, output, indent=2)
    client.drop_database(DB_NAME)
    client.close()
    print(f"\nDetailed results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
