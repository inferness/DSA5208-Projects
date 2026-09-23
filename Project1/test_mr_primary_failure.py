import os
import sys
import time
import uuid
import subprocess
from pymongo import MongoClient
from pymongo.read_preferences import Secondary
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern

# Replica set connection URI inside the docker network
URI = os.getenv(
    "MONGODB_URI",
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,mongo4:27017,mongo5:27017/?replicaSet=rs0"
)

DB_NAME = "mr_failure_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 10
PRIMARY_CONTAINER = "mongo1"

FAST_PREF = Secondary(tag_sets=[{'tier': 'fast'}])
LAGGED_PREF = Secondary(tag_sets=[{'tier': 'lagged'}])

def get_docker_client():
    """Initializes docker client if available."""
    try:
        import docker
        return docker.from_env()
    except Exception:
        return None

DOCKER_CLIENT = get_docker_client()

def kill_container(container_name=PRIMARY_CONTAINER):
    """Immediately kills the container to simulate sudden hardware/OS crash."""
    if DOCKER_CLIENT:
        try:
            c = DOCKER_CLIENT.containers.get(container_name)
            c.kill()
            return
        except Exception:
            pass
    subprocess.run(
        ["docker", "kill", container_name],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def start_container(container_name=PRIMARY_CONTAINER):
    """Starts the specified container."""
    if DOCKER_CLIENT:
        try:
            c = DOCKER_CLIENT.containers.get(container_name)
            c.start()
            return
        except Exception:
            pass
    subprocess.run(
        ["docker", "start", container_name],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def ensure_mongo1_primary(cluster_client, timeout=90):
    """
    Ensures mongo1 is running, connected, and has reclaimed PRIMARY status.
    Uses priority-assisted step-down if another node temporarily holds PRIMARY.
    """
    start_time = time.time()
    start_container(PRIMARY_CONTAINER)
    last_stepdown_time = 0

    while time.time() - start_time < timeout:
        status = None
        try:
            status = cluster_client.admin.command("replSetGetStatus")
        except Exception:
            try:
                m1_client = MongoClient("mongodb://mongo1:27017/?directConnection=true", serverSelectionTimeoutMS=2000)
                status = m1_client.admin.command("replSetGetStatus")
            except Exception:
                pass

        if status and "members" in status:
            m1_info = next((m for m in status["members"] if PRIMARY_CONTAINER in m["name"]), None)
            if m1_info:
                state = m1_info.get("stateStr")
                if state == "PRIMARY":
                    ensure_sync_source(PRIMARY_CONTAINER)
                    return True
                if state == "SECONDARY":
                    now = time.time()
                    if now - last_stepdown_time > 5:
                        curr_pri = next((m["name"] for m in status["members"] if m.get("stateStr") == "PRIMARY"), None)
                        if curr_pri and PRIMARY_CONTAINER not in curr_pri:
                            try:
                                temp_pri_client = MongoClient(
                                    f"mongodb://{curr_pri}/?directConnection=true",
                                    serverSelectionTimeoutMS=2000
                                )
                                temp_pri_client.admin.command("replSetStepDown", 15)
                                last_stepdown_time = now
                            except Exception:
                                pass
        time.sleep(1)

    raise RuntimeError(f"Timed out after {timeout}s waiting for {PRIMARY_CONTAINER} to become PRIMARY!")

def ensure_sync_source(primary_name=PRIMARY_CONTAINER):
    """
    Ensures that mongo5 (the lagged secondary) has primary_name set as its syncSourceHost.
    Because chainingAllowed is false, when mongo1 restarts, mongo5 may temporarily
    lose its sync source or back off due to sync source blacklisting.
    """
    try:
        with MongoClient("mongodb://mongo5:27017/?directConnection=true", serverSelectionTimeoutMS=2000) as m5_client:
            status5 = m5_client.admin.command("replSetGetStatus")
            sync_source = status5.get("syncSourceHost", "")
            if f"{primary_name}:27017" not in sync_source:
                try:
                    m5_client.admin.command("replSetSyncFrom", f"{primary_name}:27017")
                except Exception:
                    pass
    except Exception:
        pass

def reapply_network_delay():
    """
    Programmatically re-applies the 200ms network delay to traffic going from mongo1 to mongo5.
    This is required because 'docker start mongo1' resets the container's network namespace,
    wiping out any previously established tc rules.
    """
    try:
        mongo5_ip = None
        if DOCKER_CLIENT:
            try:
                m5 = DOCKER_CLIENT.containers.get("mongo5")
                networks = m5.attrs.get("NetworkSettings", {}).get("Networks", {})
                for net_info in networks.values():
                    if net_info.get("IPAddress"):
                        mongo5_ip = net_info["IPAddress"]
                        break
            except Exception:
                pass
                
        if not mongo5_ip:
            import socket
            try:
                mongo5_ip = socket.gethostbyname("mongo5")
            except Exception:
                pass

        if not mongo5_ip:
            print("  [Warning] Could not resolve mongo5 IP to reapply delay.")
            return

        tc_cmd = (
            f"tc qdisc del dev eth0 root 2>/dev/null || true; "
            f"tc qdisc add dev eth0 root handle 1: prio && "
            f"tc qdisc add dev eth0 parent 1:3 handle 30: netem delay 200ms && "
            f"tc filter add dev eth0 protocol ip parent 1:0 u32 match ip dst {mongo5_ip} flowid 1:3"
        )

        if DOCKER_CLIENT:
            DOCKER_CLIENT.containers.run(
                "nicolaka/netshoot",
                command=["sh", "-c", tc_cmd],
                network_mode=f"container:{PRIMARY_CONTAINER}",
                cap_add=["NET_ADMIN"],
                remove=True
            )
            return

        # Fallback to subprocess CLI if DOCKER_CLIENT is None
        subprocess.run(
            [f"docker run --rm --cap-add=NET_ADMIN --net=container:{PRIMARY_CONTAINER} nicolaka/netshoot sh -c '{tc_cmd}'"],
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"  [Warning] Failed to reapply network delay: {e}")

def cleanup_db(client):
    """Drops the test collection/database to ensure clean slate."""
    try:
        client.drop_database(DB_NAME)
    except Exception:
        pass

def run_iteration(writer_client, reader_client, cluster_client, write_concern, read_concern, iter_num, total_iters):
    doc_id = str(uuid.uuid4())
    writer_db = writer_client[DB_NAME]
    reader_db = reader_client[DB_NAME]

    # Pre-flight check
    ensure_mongo1_primary(cluster_client)

    # --- WRITER ACTOR ---
    coll_write_init = writer_db.get_collection(COLL_NAME, write_concern=WriteConcern(w="majority", wtimeout=5000))
    coll_lagged_check = writer_db.get_collection(COLL_NAME, read_concern=ReadConcern("local"), read_preference=LAGGED_PREF)
    coll_write_v2 = writer_db.get_collection(COLL_NAME, write_concern=write_concern)

    # Step 1: Write v1
    coll_write_init.insert_one({"_id": doc_id, "v": 1, "created_at": time.time()})

    # Step 2: Ensure v1 is on lagged node
    timeout = 15.0
    poll_start = time.time()
    slow_synced_v1 = False
    while time.time() - poll_start < timeout:
        doc = coll_lagged_check.find_one({"_id": doc_id})
        if doc and doc.get("v") == 1:
            slow_synced_v1 = True
            break
        if time.time() - poll_start > 2.0 and int((time.time() - poll_start) * 2) % 4 == 0:
            ensure_sync_source(PRIMARY_CONTAINER)
        time.sleep(0.05)
    
    if not slow_synced_v1:
        raise RuntimeError(f"Lagged node timed out waiting for v1 sync after {timeout}s!")

    # Step 3: Write v2
    try:
        coll_write_v2.update_one({"_id": doc_id}, {"$set": {"v": 2, "updated_at": time.time()}})
    except Exception as e:
        print(f"\n  --- [Repetition {iter_num}/{total_iters}] ---")
        print(f"    v2 Write:           FAILED ({type(e).__name__}: {e})")
        print(f"    Outcome:            INVALID PRECONDITION (The write itself failed)")
        return {
            "iter": iter_num,
            "fast_detect_ms": 0.0,
            "slow_read_ms": 0.0,
            "kill_ms": 0.0,
            "outcome": "INVALID PRECONDITION",
            "err_detail": f"v2 write failed: {type(e).__name__}",
            "observed_v": None,
            "recov_sec": 0.0
        }
        
    # --- READER ACTOR ---
    coll_fast_sec = reader_db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=FAST_PREF)
    coll_lagged_test = reader_db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=LAGGED_PREF)

    with reader_client.start_session(causal_consistency=True) as reader_session:
        # Read #1: Detect v2 on Fast Secondary
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
            print(f"\n  --- [Repetition {iter_num}/{total_iters}] ---")
            print(f"    Fast Sec Detect:    Timed out after 5.0s (v2 not observed)")
            print(f"    Outcome:            INVALID PRECONDITION (Fast secondary didn't see v2)")
            return {
                "iter": iter_num,
                "fast_detect_ms": time_fast_detect_v2_ms,
                "slow_read_ms": 0.0,
                "kill_ms": 0.0,
                "outcome": "INVALID PRECONDITION",
                "err_detail": "Fast sec didn't see v2",
                "observed_v": None,
                "recov_sec": 0.0
            }

        # Primary Crash!
        t_kill_start = time.perf_counter()
        kill_container(PRIMARY_CONTAINER)
        t_kill_end = time.perf_counter()
        kill_ms = (t_kill_end - t_kill_start) * 1000

        # Read #2: Lagged Secondary
        t_slow_start = time.perf_counter()
        doc_slow = None
        outcome = "ERROR"
        err_detail = None
        observed_val = None
        try:
            doc_slow = coll_lagged_test.find_one({"_id": doc_id}, session=reader_session)
            observed_val = doc_slow.get("v") if doc_slow else None
            
            if observed_val == 2:
                outcome = "PASS"
            elif observed_val == 1:
                outcome = "VIOLATION"
            else:
                outcome = "ERROR"
                err_detail = f"Observed unexpected v={observed_val}"
                
        except Exception as e:
            outcome = "ERROR"
            err_detail = type(e).__name__
            
        t_slow_end = time.perf_counter()
        slow_read_ms = (t_slow_end - t_slow_start) * 1000
        
    # Cluster Recovery
    t_recov_start = time.perf_counter()
    ensure_mongo1_primary(cluster_client)
    reapply_network_delay()
    ensure_sync_source(PRIMARY_CONTAINER)
    t_recov_end = time.perf_counter()
    recov_sec = t_recov_end - t_recov_start
    
    out_label = outcome if outcome != "ERROR" else f"ERROR({err_detail})"
    print(f"\n  --- [Repetition {iter_num}/{total_iters}] ---")
    print(f"    Fast Sec Detect v2: {time_fast_detect_v2_ms:7.2f} ms")
    print(f"    Kill Primary:       {PRIMARY_CONTAINER} killed in {kill_ms:.1f} ms")
    print(f"    Lagged Sec Read:    {slow_read_ms:7.2f} ms | Outcome: {out_label:<12} (Observed v={observed_val})")
    print(f"    Cluster Recovery:   Restored {PRIMARY_CONTAINER} and delay in {recov_sec:.1f}s")
    
    return {
        "iter": iter_num,
        "fast_detect_ms": time_fast_detect_v2_ms,
        "slow_read_ms": slow_read_ms,
        "kill_ms": kill_ms,
        "outcome": outcome,
        "err_detail": err_detail,
        "observed_v": observed_val,
        "recov_sec": recov_sec
    }

def test_scenario(writer_client, reader_client, cluster_client, name, write_concern, read_concern, iterations=NUM_ITERATIONS):
    print(f"\n" + "=" * 90)
    print(f"Scenario: {name} ({iterations} repetitions)")
    print("=" * 90)

    # Clean collection before scenario
    writer_client[DB_NAME][COLL_NAME].drop()

    results = []
    for i in range(1, iterations + 1):
        res = run_iteration(writer_client, reader_client, cluster_client, write_concern, read_concern, i, iterations)
        results.append(res)

    # Consistency statistics strictly from PASS + VIOLATION
    valid_mr_comparisons = [r for r in results if r["outcome"] in ("PASS", "VIOLATION")]
    passes = sum(1 for r in results if r["outcome"] == "PASS")
    violations = sum(1 for r in results if r["outcome"] == "VIOLATION")
    errors = sum(1 for r in results if r["outcome"] == "ERROR")
    invalid_preconditions = sum(1 for r in results if r["outcome"] == "INVALID PRECONDITION")

    valid_count = len(valid_mr_comparisons)
    pass_rate = (passes / valid_count * 100) if valid_count > 0 else 0.0

    # Lagged-secondary read latency strictly from valid comparisons (PASS + VIOLATION)
    avg_read_sec = (sum(r["slow_read_ms"] for r in valid_mr_comparisons) / valid_count) if valid_count > 0 else 0.0

    # Fast secondary detect latency from trials that observed v2
    fast_detected_trials = [r for r in results if r["outcome"] in ("PASS", "VIOLATION", "ERROR")]
    avg_fast_detect = (sum(r["fast_detect_ms"] for r in fast_detected_trials) / len(fast_detected_trials)) if fast_detected_trials else 0.0

    print("\n" + "-" * 90)
    print(f"  Scenario Summary: {name}")
    print(f"    Trials:                     {len(results)}")
    print(f"    Valid MR comparisons:       {valid_count} (Pass + Violation)")
    print(f"    Pass:                       {passes}")
    print(f"    Violation:                  {violations}")
    print(f"    Errors:                     {errors}")
    print(f"    Invalid precondition:       {invalid_preconditions}")
    if valid_count > 0:
        print(f"    MR Pass Rate:               {pass_rate:5.1f}% ({passes}/{valid_count})")
    else:
        print(f"    MR Pass Rate:               N/A (no valid MR comparisons)")
    print(f"    Avg Fast Detect:            {avg_fast_detect:8.2f} ms")
    print(f"    Avg Lagged Secondary Read:  {avg_read_sec:8.2f} ms (valid comparisons only)")
    print("-" * 90)

    return {
        "scenario": name,
        "trials": len(results),
        "valid_count": valid_count,
        "passes": passes,
        "violations": violations,
        "errors": errors,
        "invalid_preconditions": invalid_preconditions,
        "pass_rate": pass_rate,
        "avg_fast_detect": avg_fast_detect,
        "avg_read_sec": avg_read_sec,
        "rep_details": results
    }

def print_summary_table(all_results):
    print("\n" + "=" * 115)
    print("                    MONOTONIC READS UNDER PRIMARY FAILURE BENCHMARK SUMMARY")
    print("=" * 115)
    header = (
        f"{'Scenario':<30} | {'Trials':<6} | {'Valid':<5} | {'Pass':<5} | "
        f"{'Viol':<5} | {'MR Pass Rate':<12} | {'Error':<5} | {'Invalid':<7} | {'Avg Fast':<10} | {'Avg Lagged':<10}"
    )
    print(header)
    print("-" * 115)
    for r in all_results:
        rate_str = f"{r['pass_rate']:5.1f}%" if r["valid_count"] > 0 else "N/A"
        line = (
            f"{r['scenario']:<30} | "
            f"{r['trials']:>6} | "
            f"{r['valid_count']:>5} | "
            f"{r['passes']:>5} | "
            f"{r['violations']:>5} | "
            f"{rate_str:>12} | "
            f"{r['errors']:>5} | "
            f"{r['invalid_preconditions']:>7} | "
            f"{r['avg_fast_detect']:>7.2f} ms | "
            f"{r['avg_read_sec']:>7.2f} ms"
        )
        print(line)
    print("=" * 115)

    print("\nOutcome Legend & Methodology:")
    print("  Valid MR Comparisons : Pass + Violation (trials where Read #1 detected v2, primary failed, and Read #2 returned data).")
    print("  Pass                 : Read #2 observed v2 (Monotonic Reads preserved).")
    print("  Violation            : Read #2 observed v1 (Monotonic Reads violated - read went backwards).")
    print("  Error                : Read #2 operation failed during failover (exception/network error).")
    print("  Invalid Precondition : Trial precondition failed (v2 write failed or Fast Secondary timed out before seeing v2).")
    print("  MR Pass Rate         : Pass / Valid MR comparisons (Pass / (Pass + Violation)).")
    print("  Avg Lagged Read      : Average latency calculated strictly from valid MR comparisons (excluding failed operations).")
    print("=" * 115)

def main():
    print("=" * 80)
    print("MongoDB Monotonic Reads (MR) Benchmark Under Primary Node Failure")
    print("=" * 80)
    print(f"Iterations per scenario: {NUM_ITERATIONS}")
    print("Connecting to MongoDB Replica Set...")

    cluster_client = MongoClient(URI, serverSelectionTimeoutMS=60000, heartbeatFrequencyMS=500)
    writer_client = MongoClient(URI, serverSelectionTimeoutMS=60000, heartbeatFrequencyMS=500)
    reader_client = MongoClient(URI, serverSelectionTimeoutMS=60000, heartbeatFrequencyMS=500)
    
    cluster_client.admin.command('ping')
    writer_client.admin.command('ping')
    reader_client.admin.command('ping')
    print("Connected clients successfully!")

    print(f"Ensuring initial state: {PRIMARY_CONTAINER} is PRIMARY...")
    ensure_mongo1_primary(cluster_client)
    reapply_network_delay()
    print("Initial state verified and network delay applied.")

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
            cluster_client,
            name=tc["name"],
            write_concern=tc["wc"],
            read_concern=tc["rc"],
            iterations=NUM_ITERATIONS
        )
        all_results.append(res)

    print_summary_table(all_results)
    cleanup_db(writer_client)
    print("\nCleaned up all test databases. Test complete.\n")

if __name__ == "__main__":
    main()
