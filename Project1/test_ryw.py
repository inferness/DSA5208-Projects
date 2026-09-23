import time
import uuid
from pymongo import MongoClient, ReadPreference
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern

# Replica set connection URI inside the docker network
URI = (
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,mongo4:27017,mongo5:27017/"
    "?replicaSet=rs0"
)

DB_NAME = "ryw_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 50  # Configurable number of repetitions per case

def cleanup_db(client):
    """Drops the test collection/database to ensure clean slate."""
    client.drop_database(DB_NAME)

def run_iteration(client, write_concern, read_concern, read_pref):
    """Runs a single write -> immediate read cycle without causal consistency."""
    db = client[DB_NAME]
    doc_id = str(uuid.uuid4())
    payload = {
        "_id": doc_id,
        "written_at": time.time(),
        "status": "committed"
    }

    with client.start_session(causal_consistency=False) as session:
        coll_write = db.get_collection(COLL_NAME, write_concern=write_concern)
        coll_read = db.get_collection(COLL_NAME, read_concern=read_concern)

        # WRITE
        t_write_start = time.perf_counter()
        coll_write.insert_one(payload, session=session)
        t_write_end = time.perf_counter()
        write_ms = (t_write_end - t_write_start) * 1000

        # READ (immediate)
        t_read_start = time.perf_counter()
        read_doc = coll_read.with_options(read_preference=read_pref).find_one(
            {"_id": doc_id}, session=session
        )
        t_read_end = time.perf_counter()
        read_ms = (t_read_end - t_read_start) * 1000

    ryw_satisfied = (read_doc is not None and read_doc.get("_id") == doc_id)
    return write_ms, read_ms, ryw_satisfied

def test_scenario(client, name, write_concern, read_concern, read_pref, iterations=NUM_ITERATIONS):
    target_node = "PRIMARY" if read_pref == ReadPreference.PRIMARY else "SECONDARY"
    print(f"\nRunning: [{name} -> Read from {target_node}] ({iterations} repetitions)...")

    # Clean collection before every scenario
    client[DB_NAME][COLL_NAME].drop()

    write_latencies = []
    read_latencies = []
    success_count = 0

    t_scenario_start = time.perf_counter()

    for _ in range(iterations):
        w_ms, r_ms, success = run_iteration(client, write_concern, read_concern, read_pref)
        write_latencies.append(w_ms)
        read_latencies.append(r_ms)
        if success:
            success_count += 1

    t_scenario_end = time.perf_counter()
    total_scenario_ms = (t_scenario_end - t_scenario_start) * 1000

    success_rate = (success_count / iterations) * 100
    avg_write = sum(write_latencies) / len(write_latencies)
    avg_read = sum(read_latencies) / len(read_latencies)

    print(f"  RYW Success Rate:  {success_rate:6.1f}% ({success_count}/{iterations})")
    print(f"  Avg Write Latency: {avg_write:6.3f} ms (Min: {min(write_latencies):.2f}, Max: {max(write_latencies):.2f})")
    print(f"  Avg Read Latency:  {avg_read:6.3f} ms (Min: {min(read_latencies):.2f}, Max: {max(read_latencies):.2f})")
    print(f"  Total Duration:    {total_scenario_ms:6.2f} ms")

    return {
        "scenario": name,
        "target": target_node,
        "success_rate": success_rate,
        "avg_write": avg_write,
        "avg_read": avg_read,
        "total_ms": total_scenario_ms
    }

def print_summary_table(results):
    print("\n" + "=" * 80)
    print("                      READ-YOUR-WRITES BENCHMARK SUMMARY")
    print("=" * 80)
    header = f"{'Case':<32} | {'Target':<9} | {'RYW Pass':<9} | {'Avg Write':<10} | {'Avg Read':<10}"
    print(header)
    print("-" * 80)
    for r in results:
        line = (
            f"{r['scenario']:<32} | "
            f"{r['target']:<9} | "
            f"{r['success_rate']:>6.1f}%   | "
            f"{r['avg_write']:>7.3f} ms | "
            f"{r['avg_read']:>7.3f} ms"
        )
        print(line)
    print("=" * 80)

def main():
    print("Connecting to MongoDB Replica Set...")
    client = MongoClient(URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("Connected successfully!")

    # Initial cleanup
    cleanup_db(client)

    test_cases = [
        {
            "name": "Case 1: W=majority, R=majority",
            "wc": WriteConcern(w="majority", wtimeout=5000),
            "rc": ReadConcern("majority")
        },
        {
            "name": "Case 2: W=majority, R=local",
            "wc": WriteConcern(w="majority", wtimeout=5000),
            "rc": ReadConcern("local")
        },
        {
            "name": "Case 3: W=1, R=majority",
            "wc": WriteConcern(w=1),
            "rc": ReadConcern("majority")
        },
        {
            "name": "Case 4: W=1, R=local",
            "wc": WriteConcern(w=1),
            "rc": ReadConcern("local")
        }
    ]

    all_results = []

    for tc in test_cases:
        # Read from Primary
        res_p = test_scenario(
            client, 
            name=tc["name"], 
            write_concern=tc["wc"], 
            read_concern=tc["rc"], 
            read_pref=ReadPreference.PRIMARY
        )
        all_results.append(res_p)

        # Read from Secondary
        res_s = test_scenario(
            client, 
            name=tc["name"], 
            write_concern=tc["wc"], 
            read_concern=tc["rc"], 
            read_pref=ReadPreference.SECONDARY
        )
        all_results.append(res_s)

    # Print clean aggregated table
    print_summary_table(all_results)

    # Final cleanup so no data is left behind
    cleanup_db(client)
    print("\nCleaned up all test databases successfully. No data left behind.")

if __name__ == "__main__":
    main()
