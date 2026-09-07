import time
import uuid
from pymongo import MongoClient
from pymongo.read_preferences import Secondary
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern

# Replica set URI including all nodes
CLUSTER_URI = (
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,mongo4:27017,mongo5:27017/"
    "?replicaSet=rs0"
)

DB_NAME = "mr_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 30  # change to 30 for full test

# Define read preferences using replica set tags
FAST_PREF = Secondary(tag_sets=[{'tier': 'fast'}])
LAGGED_PREF = Secondary(tag_sets=[{'tier': 'lagged'}])

def cleanup_db(client):
    """Drops the test collection/database to ensure clean slate."""
    client.drop_database(DB_NAME)

def run_mr_iteration(client, write_concern, read_concern, causal_consistency=False):
    """
    Executes one iteration of the Monotonic Reads test protocol using a single client.
    """
    db = client[DB_NAME]
    doc_id = str(uuid.uuid4())

    with client.start_session(causal_consistency=causal_consistency) as session:
        # Collections configured with specific read/write preferences
        coll_write_init = db.get_collection(COLL_NAME, write_concern=WriteConcern(w="majority", wtimeout=5000))
        coll_lagged_check = db.get_collection(COLL_NAME, read_concern=ReadConcern("local"), read_preference=LAGGED_PREF)
        coll_write_v2 = db.get_collection(COLL_NAME, write_concern=write_concern)
        coll_fast_sec = db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=FAST_PREF)
        coll_lagged_test = db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=LAGGED_PREF)

        # Step 1: Write v1 with majority to ensure it is durable before baseline sync
        coll_write_init.insert_one({"_id": doc_id, "v": 1, "created_at": time.time()}, session=session)

        # Step 2: Poll slow node (tier: lagged) until v1 is observed
        t_sync_start = time.perf_counter()
        slow_synced_v1 = False
        timeout = 10.0  # Allow enough time for replication delay
        poll_start = time.time()

        while time.time() - poll_start < timeout:
            doc = coll_lagged_check.find_one({"_id": doc_id}, session=session)
            if doc and doc.get("v") == 1:
                slow_synced_v1 = True
                break
            time.sleep(0.05)

        t_sync_end = time.perf_counter()
        time_slow_sync_v1_ms = (t_sync_end - t_sync_start) * 1000

        if not slow_synced_v1:
            raise RuntimeError(f"Lagged node timed out waiting for v1 sync after {timeout}s!")

        # Step 3: Immediately write v2 with the test's designated write concern
        coll_write_v2.update_one({"_id": doc_id}, {"$set": {"v": 2, "updated_at": time.time()}}, session=session)

        # Step 4: Poll fast secondary pool (tier: fast) until v2 is detected
        t_detect_start = time.perf_counter()
        fast_detected_v2 = False
        
        poll_fast_start = time.time()
        while time.time() - poll_fast_start < 5.0:
            doc = coll_fast_sec.find_one({"_id": doc_id}, session=session)
            if doc and doc.get("v") == 2:
                fast_detected_v2 = True
                break
            time.sleep(0.005)

        t_detect_end = time.perf_counter()
        time_fast_detect_v2_ms = (t_detect_end - t_detect_start) * 1000

        if not fast_detected_v2:
            return {
                "time_slow_sync_v1_ms": time_slow_sync_v1_ms,
                "time_fast_detect_v2_ms": time_fast_detect_v2_ms,
                "fast_saw_v2": False,
                "violation": False,
                "observed_v": None
            }

        # Step 5: Immediately read from the slow node (tier: lagged) using the test's read concern
        doc_slow = coll_lagged_test.find_one({"_id": doc_id}, session=session)
        observed_val = doc_slow.get("v") if doc_slow else None

        # VIOLATION if slow node returns stale state (v1) after fast node has already returned v2!
        violation = (observed_val == 1)

        return {
            "time_slow_sync_v1_ms": time_slow_sync_v1_ms,
            "time_fast_detect_v2_ms": time_fast_detect_v2_ms,
            "fast_saw_v2": True,
            "violation": violation,
            "observed_v": observed_val
        }

def test_scenario(client, name, write_concern, read_concern, causal_consistency=False):
    print(f"\nRunning: [{name}] ({NUM_ITERATIONS} repetitions)...")

    violations = 0
    valid_comparisons = 0
    times_slow_sync = []
    times_fast_detect = []

    t_start = time.perf_counter()

    for i in range(NUM_ITERATIONS):
        res = run_mr_iteration(client, write_concern, read_concern, causal_consistency)
        times_slow_sync.append(res["time_slow_sync_v1_ms"])
        times_fast_detect.append(res["time_fast_detect_v2_ms"])

        if res["fast_saw_v2"]:
            valid_comparisons += 1
            if res["violation"]:
                violations += 1

        print(f"\r  Progress: {i + 1}/{NUM_ITERATIONS} iterations complete", end="", flush=True)

    t_end = time.perf_counter()
    total_duration_s = t_end - t_start

    success_rate = ((valid_comparisons - violations) / valid_comparisons * 100) if valid_comparisons > 0 else 0.0
    avg_slow_sync = sum(times_slow_sync) / len(times_slow_sync) if times_slow_sync else 0
    avg_fast_detect = sum(times_fast_detect) / len(times_fast_detect) if times_fast_detect else 0

    print(f"\n  MR Pass Rate:        {success_rate:>6.1f}% ({valid_comparisons - violations}/{valid_comparisons})")
    print(f"  MR Violations:       {violations:>6d}")
    print(f"  Avg Slow Sync v1:    {avg_slow_sync:>7.2f} ms")
    print(f"  Avg Fast Detect v2:  {avg_fast_detect:>7.2f} ms")
    print(f"  Total Duration:      {total_duration_s:>7.2f} s")

    return {
        "scenario": name,
        "success_rate": success_rate,
        "violations": violations,
        "valid_count": valid_comparisons,
        "avg_slow_sync": avg_slow_sync,
        "avg_fast_detect": avg_fast_detect,
    }

def print_summary_table(results):
    print("\n" + "=" * 90)
    print("                      MONOTONIC READS BENCHMARK SUMMARY (SINGLE CLIENT)")
    print("=" * 90)
    print(f"{'Scenario':<34} | {'MR Pass':<8} | {'Violations':<10} | {'Avg Slow Sync':<15} | {'Avg Fast Detect':<15}")
    print("-" * 90)
    for r in results:
        line = (
            f"{r['scenario']:<34} | "
            f"{r['success_rate']:>6.1f}%  | "
            f"{r['violations']:>10d} | "
            f"{r['avg_slow_sync']:>11.2f} ms | "
            f"{r['avg_fast_detect']:>11.2f} ms"
        )
        print(line)
    print("=" * 90)

def main():
    print("Connecting to MongoDB Replica Set using single client...")
    client = MongoClient(CLUSTER_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("Connected successfully to cluster!")

    cleanup_db(client)

    test_cases = [
        {
            "name": "Case 1: W=majority, R=majority",
            "wc": WriteConcern(w="majority", wtimeout=5000),
            "rc": ReadConcern("majority")
        },
        {
            "name": "Case 2: W=1, R=majority",
            "wc": WriteConcern(w=1),
            "rc": ReadConcern("majority")
        },
        {
            "name": "Case 3: W=1, R=local",
            "wc": WriteConcern(w=1),
            "rc": ReadConcern("local")
        },
        {
            "name": "Case 4: W=majority, R=local",
            "wc": WriteConcern(w="majority", wtimeout=5000),
            "rc": ReadConcern("local")
        }
    ]

    all_results = []
    for tc in test_cases:
        res = test_scenario(
            client,
            name=tc["name"],
            write_concern=tc["wc"],
            read_concern=tc["rc"],
            causal_consistency=False  # You can toggle this to True to see causal consistency in action
        )
        all_results.append(res)

    print_summary_table(all_results)
    cleanup_db(client)
    print("\nCleaned up all test databases successfully.")

if __name__ == "__main__":
    main()
