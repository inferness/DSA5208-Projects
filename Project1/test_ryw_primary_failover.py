import os
import sys
import time
import uuid
import subprocess
from pymongo import MongoClient, ReadPreference
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern

# Replica set connection URI inside the docker network
URI = os.getenv(
    "MONGODB_URI",
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,mongo4:27017,mongo5:27017/?replicaSet=rs0"
)

DB_NAME = "ryw_failover_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 10  # Default repetitions per scenario
PRIMARY_CONTAINER = "mongo1"
SERVER_SELECTION_TIMEOUT_MS = 60000  # 60s to allow failover & election to finish
CAUSAL_CONSISTENCY = False
MAX_TIME_MS = int(os.getenv("MAX_TIME_MS", "30000"))  # 30,000 ms = 30 seconds


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
        except Exception as e:
            print(f"  [Docker SDK Warning] Failed to kill {container_name}: {e}")
    # Fallback to docker CLI
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
        except Exception as e:
            print(f"  [Docker SDK Warning] Failed to start {container_name}: {e}")
    # Fallback to docker CLI
    subprocess.run(
        ["docker", "start", container_name],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


RECOVERY_TIMEOUT_SEC = int(os.getenv("RECOVERY_TIMEOUT_SEC", "90"))  # 90s to accommodate journal replay and election timers


def ensure_mongo1_primary(cluster_client, timeout=RECOVERY_TIMEOUT_SEC):
    """
    Ensures mongo1 is running, connected, and has reclaimed PRIMARY status.
    Uses priority-assisted step-down if another node temporarily holds PRIMARY.
    """
    start_time = time.time()
    # First, make sure container is started
    start_container(PRIMARY_CONTAINER)

    last_stepdown_time = 0

    while time.time() - start_time < timeout:
        status = None
        # Try checking status via cluster_client first (online surviving nodes), fallback to direct mongo1
        try:
            status = cluster_client.admin.command("replSetGetStatus")
        except Exception:
            try:
                with MongoClient("mongodb://mongo1:27017/?directConnection=true", serverSelectionTimeoutMS=2000) as m1_client:
                    status = m1_client.admin.command("replSetGetStatus")
            except Exception:
                pass

        if status and "members" in status:
            m1_info = next((m for m in status["members"] if PRIMARY_CONTAINER in m["name"]), None)
            if m1_info:
                state = m1_info.get("stateStr")
                if state == "PRIMARY":
                    return True
                if state == "SECONDARY":
                    # If another node is currently PRIMARY, ask it to step down so mongo1 (priority 2) wins
                    now = time.time()
                    # Throttle step-down requests to once every 5 seconds to allow election to settle
                    if now - last_stepdown_time > 5:
                        curr_pri = next((m["name"] for m in status["members"] if m.get("stateStr") == "PRIMARY"), None)
                        if curr_pri and PRIMARY_CONTAINER not in curr_pri:
                            try:
                                with MongoClient(
                                    f"mongodb://{curr_pri}/?directConnection=true",
                                    serverSelectionTimeoutMS=2000
                                ) as temp_pri_client:
                                    temp_pri_client.admin.command("replSetStepDown", 15)
                                last_stepdown_time = now
                            except Exception:
                                pass
        time.sleep(1)

    raise RuntimeError(f"Timed out after {timeout}s waiting for {PRIMARY_CONTAINER} to become PRIMARY!")


def cleanup_db(client):
    """Drops the test database."""
    try:
        client.drop_database(DB_NAME)
    except Exception:
        pass


def run_failover_iteration(cluster_client, write_concern, read_concern, iter_num, total_iters, causal_consistency=CAUSAL_CONSISTENCY):
    """
    Executes one failover test cycle:
      1. Verify mongo1 is PRIMARY.
      2. Write document to PRIMARY (within causally consistent session).
      3. Abruptly kill mongo1 (primary crash).
      4. Measure immediate read on surviving SECONDARY (availability & consistency, within session).
      5. Measure read on PRIMARY (election detection & request resolution, within session).
      6. Restart mongo1 and restore it as PRIMARY.
    """
    db = cluster_client[DB_NAME]
    doc_id = str(uuid.uuid4())
    payload = {
        "_id": doc_id,
        "written_at": time.time(),
        "status": "committed",
        "iteration": iter_num
    }

    # Step 1: Pre-flight check - mongo1 must be PRIMARY
    ensure_mongo1_primary(cluster_client)

    # Wrap the write -> kill -> reads in a single client session to enforce causal consistency (RYW)
    with cluster_client.start_session(causal_consistency=causal_consistency) as session:
        # Step 2: Write document
        coll_write = db.get_collection(COLL_NAME, write_concern=write_concern)
        t_write_start = time.perf_counter()
        write_err = None
        try:
            coll_write.insert_one(payload, session=session)
        except Exception as e:
            write_err = type(e).__name__
        t_write_end = time.perf_counter()
        write_ms = (t_write_end - t_write_start) * 1000

        # Step 3: Abruptly Kill Primary (mongo1)
        t_fail_start = time.perf_counter()
        kill_container(PRIMARY_CONTAINER)
        t_fail_end = time.perf_counter()
        kill_ms = (t_fail_end - t_fail_start) * 1000

        # Step 4: Immediate Read strictly from surviving SECONDARY
        coll_read_sec = db.get_collection(
            COLL_NAME,
            read_concern=read_concern
        ).with_options(read_preference=ReadPreference.SECONDARY)

        t_sec_start = time.perf_counter()
        doc_sec = None
        sec_outcome = "ERROR"
        sec_err_detail = None
        try:
            doc_sec = coll_read_sec.find_one(
                {"_id": doc_id},
                session=session,
                max_time_ms=MAX_TIME_MS
            )
            if doc_sec is not None and doc_sec.get("_id") == doc_id:
                sec_outcome = "PASS"
            else:
                sec_outcome = "VIOLATION"
        except Exception as e:
            sec_outcome = "ERROR"
            sec_err_detail = type(e).__name__
        t_sec_end = time.perf_counter()
        read_sec_ms = (t_sec_end - t_sec_start) * 1000

        # Step 5: Read from PRIMARY (waiting for election & resolution)
        # PyMongo server selection loop will wait until a new primary is elected
        coll_read_pri = db.get_collection(
            COLL_NAME,
            read_concern=read_concern
        ).with_options(read_preference=ReadPreference.PRIMARY)

        t_pri_start = time.perf_counter()
        doc_pri = None
        pri_outcome = "ERROR"
        pri_err_detail = None
        elected_primary = "unknown"
        try:
            doc_pri = coll_read_pri.find_one(
                {"_id": doc_id},
                session=session,
                max_time_ms=MAX_TIME_MS
            )
            elected_primary = cluster_client.primary[0] if cluster_client.primary else "unknown"
            if doc_pri is not None and doc_pri.get("_id") == doc_id:
                pri_outcome = "PASS"
            else:
                pri_outcome = "VIOLATION"
        except Exception as e:
            pri_outcome = "ERROR"
            pri_err_detail = type(e).__name__
        t_pri_end = time.perf_counter()
        read_pri_ms = (t_pri_end - t_pri_start) * 1000
        total_failover_ms = (t_pri_end - t_fail_start) * 1000

    # Step 6: Cluster Recovery & Restoration
    t_recov_start = time.perf_counter()
    ensure_mongo1_primary(cluster_client)
    t_recov_end = time.perf_counter()
    recov_sec = t_recov_end - t_recov_start

    # Format status labels for console logging
    sec_label = sec_outcome if sec_outcome != "ERROR" else f"ERROR({sec_err_detail})"
    pri_label = pri_outcome if pri_outcome != "ERROR" else f"ERROR({pri_err_detail})"
    write_status = "OK" if write_err is None else f"ERROR({write_err})"

    print(f"\n  --- [Repetition {iter_num}/{total_iters}] ---")
    print(f"    Write:             {write_ms:7.2f} ms | Status: {write_status}")
    print(f"    Kill Primary:      {PRIMARY_CONTAINER} killed (SIGKILL) in {kill_ms:.1f} ms")
    print(f"    Secondary Read:    {read_sec_ms:7.2f} ms | Outcome: {sec_label:<12} (Target: Surviving SECONDARY)")
    print(f"    Primary Read:      {read_pri_ms:7.2f} ms (Total from crash: {total_failover_ms:7.2f} ms) | Elected: {elected_primary:<6} | Outcome: {pri_label:<12}")
    print(f"    Cluster Recovery:  Restored {PRIMARY_CONTAINER} as PRIMARY in {recov_sec:.1f}s")

    return {
        "iter": iter_num,
        "write_ms": write_ms,
        "write_status": write_status,
        "read_sec_ms": read_sec_ms,
        "sec_outcome": sec_outcome,
        "sec_err_detail": sec_err_detail,
        "read_pri_ms": read_pri_ms,
        "total_failover_ms": total_failover_ms,
        "pri_outcome": pri_outcome,
        "pri_err_detail": pri_err_detail,
        "elected_primary": elected_primary,
        "recov_sec": recov_sec
    }


def test_scenario(cluster_client, name, write_concern, read_concern, iterations=NUM_ITERATIONS, causal_consistency=CAUSAL_CONSISTENCY):
    print(f"\n" + "=" * 90)
    print(f"Scenario: {name} (Causal Consistency: {causal_consistency}, {iterations} repetitions)")
    print("=" * 90)

    # Clean collection before scenario
    cluster_client[DB_NAME][COLL_NAME].drop()

    results = []
    for i in range(1, iterations + 1):
        res = run_failover_iteration(cluster_client, write_concern, read_concern, i, iterations, causal_consistency=causal_consistency)
        results.append(res)

    # Aggregations
    avg_write = sum(r["write_ms"] for r in results) / len(results)
    avg_read_sec = sum(r["read_sec_ms"] for r in results) / len(results)
    avg_read_pri = sum(r["read_pri_ms"] for r in results) / len(results)
    avg_total_failover = sum(r["total_failover_ms"] for r in results) / len(results)

    sec_pass = sum(1 for r in results if r["sec_outcome"] == "PASS")
    sec_viol = sum(1 for r in results if r["sec_outcome"] == "VIOLATION")
    sec_err = sum(1 for r in results if r["sec_outcome"] == "ERROR")

    pri_pass = sum(1 for r in results if r["pri_outcome"] == "PASS")
    pri_viol = sum(1 for r in results if r["pri_outcome"] == "VIOLATION")
    pri_err = sum(1 for r in results if r["pri_outcome"] == "ERROR")

    print("\n" + "-" * 90)
    print(f"  Scenario Summary: {name} (Causal Consistency={causal_consistency})")
    print(f"    Avg Write Latency:         {avg_write:8.2f} ms")
    print(f"    SECONDARY Read Latency:    {avg_read_sec:8.2f} ms | Outcomes: PASS={sec_pass}, VIOLATION={sec_viol}, ERROR={sec_err}")
    print(f"    PRIMARY Read / Resolution: {avg_read_pri:8.2f} ms (Total from crash: {avg_total_failover:8.2f} ms) | Outcomes: PASS={pri_pass}, VIOLATION={pri_viol}, ERROR={pri_err}")
    print("-" * 90)

    return {
        "scenario": name,
        "causal_consistency": causal_consistency,
        "avg_write": avg_write,
        "avg_read_sec": avg_read_sec,
        "sec_counts": (sec_pass, sec_viol, sec_err),
        "avg_read_pri": avg_read_pri,
        "avg_total_failover": avg_total_failover,
        "pri_counts": (pri_pass, pri_viol, pri_err),
        "iterations": iterations,
        "rep_details": results
    }


def print_summary_table(all_results, causal_consistency=CAUSAL_CONSISTENCY):
    print("\n" + "=" * 105)
    print(f"  READ-YOUR-WRITES UNDER PRIMARY FAILURE BENCHMARK SUMMARY (Causal Consistency: {causal_consistency})")
    print("=" * 105)
    header = (
        f"{'Scenario':<30} | {'Avg Write':<10} | "
        f"{'Pri Election Wait':<18} | {'Pri (P / V / E)':<16} | "
        f"{'Sec Read':<10} | {'Sec (P / V / E)':<16}"
    )
    print(header)
    print("-" * 107)
    for r in all_results:
        p_p, p_v, p_e = r["pri_counts"]
        s_p, s_v, s_e = r["sec_counts"]
        pri_str = f"{p_p} / {p_v} / {p_e}"
        sec_str = f"{s_p} / {s_v} / {s_e}"
        line = (
            f"{r['scenario']:<30} | "
            f"{r['avg_write']:>7.2f} ms | "
            f"{r['avg_total_failover']:>15.2f} ms | "
            f"{pri_str:^16} | "
            f"{r['avg_read_sec']:>7.2f} ms | "
            f"{sec_str:^16}"
        )
        print(line)
    print("=" * 107)

    print("\nOutcome Legend (Counts formatted as PASS / VIOLATION / ERROR):")
    print("  PASS      : Read completed successfully and document was found (RYW preserved).")
    print("  VIOLATION : Read completed successfully but document was absent (stale read / un-replicated write lost).")
    print("  ERROR     : Read request could not be completed (e.g. server selection timeout / connection drop).")

    print("\n" + "-" * 107)
    print("                      ITEMIZED REPETITION LOG (FOR REPORT REFERENCE)")
    print("-" * 107)
    rep_header = f"{'Scenario':<20} | {'Rep':<4} | {'Write':<10} | {'Sec Read':<9} | {'Sec Outcome':<12} | {'Pri Wait':<12} | {'Elected':<7} | {'Pri Outcome':<12}"
    print(rep_header)
    print("-" * 107)
    for r in all_results:
        for rep in r["rep_details"]:
            sec_o = rep["sec_outcome"] if rep["sec_outcome"] != "ERROR" else f"ERR({rep['sec_err_detail']})"
            pri_o = rep["pri_outcome"] if rep["pri_outcome"] != "ERROR" else f"ERR({rep['pri_err_detail']})"
            scen_short = r["scenario"].replace("majority", "maj").replace("local", "loc")
            rep_line = (
                f"{scen_short:<20} | "
                f"#{rep['iter']:<3} | "
                f"{rep['write_ms']:>7.2f} ms | "
                f"{rep['read_sec_ms']:>6.2f} ms | "
                f"{sec_o:<12} | "
                f"{rep['total_failover_ms']:>9.2f} ms | "
                f"{rep['elected_primary']:<7} | "
                f"{pri_o:<12}"
            )
            print(rep_line)
    print("=" * 107)


def main():
    print("=" * 80)
    print("MongoDB Read-Your-Writes (RYW) Benchmark Under Primary Node Failure")
    print(f"Causal Consistency: {CAUSAL_CONSISTENCY}")
    print("=" * 80)
    print(f"Cluster URI: {URI}")
    print(f"Iterations per scenario: {NUM_ITERATIONS}")
    print("Connecting to MongoDB Replica Set...")

    client = MongoClient(URI, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
    client.admin.command('ping')
    print("Connected to replica set!")

    print(f"Ensuring initial state: {PRIMARY_CONTAINER} is PRIMARY...")
    ensure_mongo1_primary(client)
    print(f"Initial state verified. {PRIMARY_CONTAINER} is PRIMARY.")

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
        res = test_scenario(
            client,
            name=tc["name"],
            write_concern=tc["wc"],
            read_concern=tc["rc"],
            iterations=NUM_ITERATIONS,
            causal_consistency=CAUSAL_CONSISTENCY
        )
        all_results.append(res)

    print_summary_table(all_results, causal_consistency=CAUSAL_CONSISTENCY)
    cleanup_db(client)
    print("\nCleaned up all test databases. Test complete.\n")


if __name__ == "__main__":
    main()
