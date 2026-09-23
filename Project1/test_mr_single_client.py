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
NUM_ITERATIONS = 30  # Standard statistical sample size (N >= 30)
CAUSAL_CONSISTENCY = True  # Set to True to enable causal consistency in the reader session

# Define read preferences using replica set tags
FAST_PREF = Secondary(tag_sets=[{'tier': 'fast'}])
LAGGED_PREF = Secondary(tag_sets=[{'tier': 'lagged'}])

def cleanup_db(client):
    """Drops the test collection/database to ensure clean slate."""
    client.drop_database(DB_NAME)

def run_mr_iteration(writer_client, reader_client, write_concern, read_concern, causal_consistency=False):
    """
    Executes one iteration of the Monotonic Reads test protocol.
    
    Architecture:
      - Writer Actor (writer_client): writes v1 and v2, completely independent of the reader.
      - Reader Actor (reader_client): uses a single ClientSession to perform:
          Read #1: Fast secondary (tier: fast) -> observes v2
          Read #2: Lagged secondary (tier: lagged) -> evaluates if monotonic read holds (v2) or violates (v1)
    """
    doc_id = str(uuid.uuid4())
    writer_db = writer_client[DB_NAME]
    reader_db = reader_client[DB_NAME]

    # --- WRITER ACTOR (Setup & Mutations) ---
    coll_write_init = writer_db.get_collection(COLL_NAME, write_concern=WriteConcern(w="majority", wtimeout=5000))
    coll_lagged_check = writer_db.get_collection(COLL_NAME, read_concern=ReadConcern("local"), read_preference=LAGGED_PREF)
    coll_write_v2 = writer_db.get_collection(COLL_NAME, write_concern=write_concern)

    # Step 1: Writer writes v1 with majority write concern
    coll_write_init.insert_one({"_id": doc_id, "v": 1, "created_at": time.time()})

    # Step 2: Writer polls slow node (tier: lagged) until baseline v1 is replicated
    # (Experimental setup only - outside of the reader session)
    timeout = 10.0
    poll_start = time.time()
    slow_synced_v1 = False
    while time.time() - poll_start < timeout:
        doc = coll_lagged_check.find_one({"_id": doc_id})
        if doc and doc.get("v") == 1:
            slow_synced_v1 = True
            break
        time.sleep(0.05)

    if not slow_synced_v1:
        raise RuntimeError(f"Lagged node timed out waiting for v1 sync after {timeout}s!")

    # Step 3: Writer updates record to v2 with designated write concern
    coll_write_v2.update_one({"_id": doc_id}, {"$set": {"v": 2, "updated_at": time.time()}})

    # --- READER ACTOR (Monotonic Reads Evaluation) ---
    # Isolated in its own reader session without any writer contamination
    coll_fast_sec = reader_db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=FAST_PREF)
    coll_lagged_test = reader_db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=LAGGED_PREF)

    with reader_client.start_session(causal_consistency=causal_consistency) as reader_session:
        # Step 4: Read #1 on Fast Secondary (tier: fast) until v2 is observed
        t_detect_start = time.perf_counter()
        fast_detected_v2 = False
        poll_fast_start = time.time()

        while time.time() - poll_fast_start < 5.0:
            doc_fast = coll_fast_sec.find_one({"_id": doc_id}, session=reader_session)
            if doc_fast and doc_fast.get("v") == 2:
                fast_detected_v2 = True
                break
            time.sleep(0.002)

        t_detect_end = time.perf_counter()
        time_fast_detect_v2_ms = (t_detect_end - t_detect_start) * 1000

        if not fast_detected_v2:
            return {
                "fast_detect_ms": time_fast_detect_v2_ms,
                "slow_read_ms": 0.0,
                "fast_saw_v2": False,
                "violation": False,
                "observed_v": None
            }

        # Step 5: Read #2 on Slow Secondary (tier: lagged) within the SAME reader session
        # Precisely measure the duration of this critical read
        t_slow_start = time.perf_counter()
        doc_slow = coll_lagged_test.find_one({"_id": doc_id}, session=reader_session)
        t_slow_end = time.perf_counter()
        slow_read_ms = (t_slow_end - t_slow_start) * 1000

        observed_val = doc_slow.get("v") if doc_slow else None

        # Monotonic Reads Violation: Read #1 observed v2, but Read #2 went backward to v1!
        violation = (observed_val == 1)

        return {
            "fast_detect_ms": time_fast_detect_v2_ms,
            "slow_read_ms": slow_read_ms,
            "fast_saw_v2": True,
            "violation": violation,
            "observed_v": observed_val
        }

def test_scenario(writer_client, reader_client, name, write_concern, read_concern, causal_consistency=False):
    mode_str = "Causal ON" if causal_consistency else "Causal OFF"
    print(f"\nRunning: [{name} | {mode_str}] ({NUM_ITERATIONS} repetitions)...")

    violations = 0
    valid_comparisons = 0
    times_fast_detect = []
    times_slow_read = []

    t_start = time.perf_counter()

    for i in range(NUM_ITERATIONS):
        res = run_mr_iteration(writer_client, reader_client, write_concern, read_concern, causal_consistency)
        times_fast_detect.append(res["fast_detect_ms"])
        times_slow_read.append(res["slow_read_ms"])

        if res["fast_saw_v2"]:
            valid_comparisons += 1
            if res["violation"]:
                violations += 1

        print(f"\r  Progress: {i + 1}/{NUM_ITERATIONS} iterations complete", end="", flush=True)

    t_end = time.perf_counter()
    total_duration_s = t_end - t_start

    success_rate = ((valid_comparisons - violations) / valid_comparisons * 100) if valid_comparisons > 0 else 0.0
    avg_fast_detect = sum(times_fast_detect) / len(times_fast_detect) if times_fast_detect else 0
    avg_slow_read = sum(times_slow_read) / len(times_slow_read) if times_slow_read else 0
    min_slow_read = min(times_slow_read) if times_slow_read else 0
    max_slow_read = max(times_slow_read) if times_slow_read else 0

    print(f"\n  MR Pass Rate:         {success_rate:>6.1f}% ({valid_comparisons - violations}/{valid_comparisons})")
    print(f"  MR Violations:        {violations:>6d}")
    print(f"  Avg Fast Detect v2:   {avg_fast_detect:>7.2f} ms")
    print(f"  Avg Slow Read Latency:{avg_slow_read:>7.2f} ms (Min: {min_slow_read:.2f}, Max: {max_slow_read:.2f})")
    print(f"  Total Duration:       {total_duration_s:>7.2f} s")

    return {
        "scenario": name,
        "mode": mode_str,
        "success_rate": success_rate,
        "violations": violations,
        "valid_count": valid_comparisons,
        "avg_fast_detect": avg_fast_detect,
        "avg_slow_read": avg_slow_read,
    }

def print_summary_table(results):
    print("\n" + "=" * 98)
    print("                      MONOTONIC READS BENCHMARK SUMMARY (ISOLATED CLIENTS)")
    print("=" * 98)
    print(f"{'Scenario':<34} | {'Mode':<10} | {'MR Pass':<8} | {'Violations':<10} | {'Avg Slow Read':<15} | {'Avg Fast Detect':<15}")
    print("-" * 98)
    for r in results:
        line = (
            f"{r['scenario']:<34} | "
            f"{r['mode']:<10} | "
            f"{r['success_rate']:>6.1f}%  | "
            f"{r['violations']:>10d} | "
            f"{r['avg_slow_read']:>11.2f} ms | "
            f"{r['avg_fast_detect']:>11.2f} ms"
        )
        print(line)
    print("=" * 98)

def main():
    print("Connecting to MongoDB Replica Set...")
    writer_client = MongoClient(CLUSTER_URI, serverSelectionTimeoutMS=5000)
    reader_client = MongoClient(CLUSTER_URI, serverSelectionTimeoutMS=5000)
    writer_client.admin.command('ping')
    reader_client.admin.command('ping')
    print("Connected successfully with isolated Writer and Reader clients!")

    cleanup_db(writer_client)

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
            writer_client,
            reader_client,
            name=tc["name"],
            write_concern=tc["wc"],
            read_concern=tc["rc"],
            causal_consistency=CAUSAL_CONSISTENCY
        )
        all_results.append(res)

    print_summary_table(all_results)
    cleanup_db(writer_client)
    print("\nCleaned up all test databases successfully.")

if __name__ == "__main__":
    main()
