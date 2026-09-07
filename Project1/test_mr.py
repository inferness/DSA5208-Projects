import time
import uuid
from pymongo import MongoClient, ReadPreference
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern

# Replica set URI (for writing to primary and reading from fast secondaries)
CLUSTER_URI = (
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,mongo4:27017/"
    "?replicaSet=rs0"
)

# Direct connection to the artificially lagged member (mongo5)
LAGGED_NODE_URI = "mongodb://mongo5:27017/?directConnection=true"

DB_NAME = "mr_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 50  # 30 repetitions per test case

def cleanup_db(client):
    """Drops the test collection/database to ensure clean slate."""
    client.drop_database(DB_NAME)

def run_mr_iteration(client_cluster, client_lagged, write_concern, read_concern):
    """
    Executes one iteration of the Monotonic Reads test protocol:
      1. Write version 1 (W1).
      2. Poll slow client (mongo5) until v1 is confirmed synced.
      3. Immediately write version 2 (W2) with given write concern.
      4. Poll fast client until v2 is observed on a fast secondary.
      5. Immediately query slow client. If it returns v1 -> Monotonic Read VIOLATION.
    """
    db_cluster = client_cluster[DB_NAME]
    db_lagged = client_lagged[DB_NAME]

    doc_id = str(uuid.uuid4())

    # Step 1: Write v1 with majority to ensure it is durable before baseline sync
    coll_write_init = db_cluster.get_collection(COLL_NAME, write_concern=WriteConcern(w="majority", wtimeout=5000))
    coll_write_init.insert_one({"_id": doc_id, "v": 1, "created_at": time.time()})

    # Step 2: Poll slow client (mongo5) until v1 is observed
    t_sync_start = time.perf_counter()
    slow_synced_v1 = False
    timeout = 10.0  # Allow enough time for mongo5 replication delay
    poll_start = time.time()

    coll_lagged_check = db_lagged.get_collection(COLL_NAME, read_concern=ReadConcern("local"))
    while time.time() - poll_start < timeout:
        doc = coll_lagged_check.find_one({"_id": doc_id})
        if doc and doc.get("v") == 1:
            slow_synced_v1 = True
            break
        time.sleep(0.05)

    t_sync_end = time.perf_counter()
    time_slow_sync_v1_ms = (t_sync_end - t_sync_start) * 1000

    if not slow_synced_v1:
        raise RuntimeError(f"Slow node (mongo5) timed out waiting for v1 sync after {timeout}s!")

    # Step 3: Immediately write v2 with the test's designated write concern
    coll_write_v2 = db_cluster.get_collection(COLL_NAME, write_concern=write_concern)
    coll_write_v2.update_one({"_id": doc_id}, {"$set": {"v": 2, "updated_at": time.time()}})

    # Step 4: Poll fast secondary pool until v2 is detected
    t_detect_start = time.perf_counter()
    fast_detected_v2 = False
    coll_fast_sec = db_cluster.get_collection(
        COLL_NAME,
        read_concern=read_concern
    ).with_options(read_preference=ReadPreference.SECONDARY)

    poll_fast_start = time.time()
    while time.time() - poll_fast_start < 5.0:
        doc = coll_fast_sec.find_one({"_id": doc_id})
        if doc and doc.get("v") == 2:
            fast_detected_v2 = True
            break
        time.sleep(0.005)

    t_detect_end = time.perf_counter()
    time_fast_detect_v2_ms = (t_detect_end - t_detect_start) * 1000

    if not fast_detected_v2:
        # Fast pool never saw v2 within timeout
        return {
            "time_slow_sync_v1_ms": time_slow_sync_v1_ms,
            "time_fast_detect_v2_ms": time_fast_detect_v2_ms,
            "fast_saw_v2": False,
            "violation": False,
            "observed_v": None
        }

    # Step 5: Immediately read from the slow node (mongo5) using the test's read concern
    # Without causal consistency, simulate an observer reading after seeing v2
    coll_lagged_test = db_lagged.get_collection(COLL_NAME, read_concern=read_concern)
    doc_slow = coll_lagged_test.find_one({"_id": doc_id})
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

def test_scenario(client_cluster, client_lagged, name, write_concern, read_concern):
    print(f"\nRunning: [{name}] ({NUM_ITERATIONS} repetitions)...")

    violations = 0
    valid_comparisons = 0
    times_slow_sync = []
    times_fast_detect = []

    t_start = time.perf_counter()

    for i in range(NUM_ITERATIONS):
        res = run_mr_iteration(client_cluster, client_lagged, write_concern, read_concern)
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
    print("                      MONOTONIC READS BENCHMARK SUMMARY")
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
    print("Connecting to MongoDB Replica Set & Lagged Node...")
    client_cluster = MongoClient(CLUSTER_URI, serverSelectionTimeoutMS=5000)
    client_cluster.admin.command('ping')

    client_lagged = MongoClient(LAGGED_NODE_URI, serverSelectionTimeoutMS=5000, directConnection=True)
    client_lagged.admin.command('ping')
    print("Connected successfully to cluster and lagged node (mongo5)!")

    cleanup_db(client_cluster)

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
            client_cluster,
            client_lagged,
            name=tc["name"],
            write_concern=tc["wc"],
            read_concern=tc["rc"]
        )
        all_results.append(res)

    print_summary_table(all_results)
    cleanup_db(client_cluster)
    print("\nCleaned up all test databases successfully.")

if __name__ == "__main__":
    main()
