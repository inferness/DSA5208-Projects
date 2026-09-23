import os
import sys
import time
import uuid
from pymongo import MongoClient, ReadPreference, monitoring
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern

# Replica set connection URI inside the docker network
URI = os.getenv(
    "MONGODB_URI",
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,mongo4:27017,mongo5:27017/?replicaSet=rs0"
)

DB_NAME = "ryw_partition_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 30  # Default 3 repetitions per scenario
MINORITY_NODES = ["mongo1:27017", "mongo2:27017"]
MAJORITY_NODES = ["mongo3:27017", "mongo4:27017", "mongo5:27017"]
SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("SERVER_SELECTION_TIMEOUT_MS", "30000"))
WRITE_TIMEOUT_MS = int(os.getenv("WRITE_TIMEOUT_MS", "10000"))
HEAL_ON_EXIT = os.getenv("HEAL_ON_EXIT", "false").lower() == "true"


class WriteTrackingListener(monitoring.CommandListener):
    """Monitors write commands to track the exact server node that fulfilled each write."""
    def __init__(self):
        self.last_write_node = "unknown"

    def started(self, event):
        pass

    def succeeded(self, event):
        if event.command_name in ("insert", "update", "delete"):
            addr = getattr(event, "address", None)
            conn = getattr(event, "connection_id", None)
            if addr and isinstance(addr, (list, tuple)):
                self.last_write_node = f"{addr[0]}:{addr[1]}"
            elif conn and isinstance(conn, (list, tuple)):
                self.last_write_node = f"{conn[0]}:{conn[1]}"
            elif addr:
                self.last_write_node = str(addr)
            elif conn:
                self.last_write_node = str(conn)

    def failed(self, event):
        addr = getattr(event, "address", None)
        conn = getattr(event, "connection_id", None)
        if addr and isinstance(addr, (list, tuple)):
            self.last_write_node = f"{addr[0]}:{addr[1]}"
        elif conn and isinstance(conn, (list, tuple)):
            self.last_write_node = f"{conn[0]}:{conn[1]}"


WRITE_LISTENER = WriteTrackingListener()


def verify_partition_topology():
    """
    Inspects the established 2-3 partition across both minority and majority sides:
      Minority (mongo1, mongo2): No writable PRIMARY.
      Majority (mongo3, mongo4, mongo5): Exactly one PRIMARY, remaining SECONDARY.
    """
    print("\n" + "=" * 95)
    print("  Verifying Established 2–3 Network Partition Topology")
    print("=" * 95)

    # 1. Check Majority Side
    majority_primary = None
    majority_secondaries = []
    for host in MAJORITY_NODES:
        try:
            client = MongoClient(f"mongodb://{host}/?directConnection=true", serverSelectionTimeoutMS=2000)
            hello = client.admin.command("hello")
            if hello.get("isWritablePrimary"):
                majority_primary = host
            elif hello.get("secondary"):
                majority_secondaries.append(host)
            client.close()
        except Exception as e:
            print(f"  [Majority Check Warning] Could not connect directly to {host}: {e}")

    # 2. Check Minority Side
    minority_primaries = []
    minority_secondaries = []
    for host in MINORITY_NODES:
        try:
            client = MongoClient(f"mongodb://{host}/?directConnection=true", serverSelectionTimeoutMS=2000)
            hello = client.admin.command("hello")
            if hello.get("isWritablePrimary"):
                minority_primaries.append(host)
            elif hello.get("secondary"):
                minority_secondaries.append(host)
            client.close()
        except Exception as e:
            print(f"  [Minority Check Warning] Could not connect directly to {host}: {e}")

    print(f"  Majority Partition ({', '.join(MAJORITY_NODES)}):")
    print(f"    - Elected PRIMARY:  {majority_primary or 'NONE (Election pending?)'}")
    print(f"    - SECONDARIES:      {', '.join(majority_secondaries)}")

    print(f"  Minority Partition ({', '.join(MINORITY_NODES)}):")
    print(f"    - PRIMARY:          {', '.join(minority_primaries) if minority_primaries else 'NONE (Headless - Expected)'}")
    print(f"    - SECONDARIES:      {', '.join(minority_secondaries)}")

    if not majority_primary:
        print("\n  [WARNING] No PRIMARY detected in majority partition! Server election may still be settling.")
    elif minority_primaries:
        print(f"\n  [WARNING] Minority partition still has a PRIMARY: {minority_primaries}! Step-down may be settling.")
    else:
        print(f"\n  ==> Topology is STABLE. Active Majority Primary: {majority_primary}\n")

    return majority_primary


def execute_read_with_node(coll, query, session, read_pref):
    """
    Executes a read query with the given ReadPreference and captures
    the document returned and the exact server node (host:port) that served it.
    """
    read_coll = coll.with_options(read_preference=read_pref)
    t_start = time.perf_counter()
    serving_node = "unknown"
    doc = None
    outcome = "ERROR"
    err_detail = None

    try:
        cursor = read_coll.find(query, session=session).limit(1)
        doc = next(cursor, None)
        if cursor.address:
            serving_node = f"{cursor.address[0]}:{cursor.address[1]}"
        if doc is not None and doc.get("_id") == query["_id"]:
            outcome = "PASS"
        else:
            outcome = "VIOLATION"
    except Exception as e:
        outcome = "ERROR"
        err_detail = type(e).__name__
    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    return {
        "outcome": outcome,
        "latency_ms": latency_ms,
        "serving_node": serving_node,
        "err_detail": err_detail,
        "doc": doc
    }


def run_trial(cluster_client, write_concern, read_concern, iter_num, total_iters):
    """
    Executes one trial within the established steady-state partition:
      1. Write unique document via replica set client.
      2. Record write latency, status, and write serving node.
      3. Immediately read PRIMARY and record serving node.
      4. Immediately read SECONDARY and record serving node.
    """
    db = cluster_client[DB_NAME]
    doc_id = str(uuid.uuid4())
    payload = {
        "_id": doc_id,
        "written_at": time.time(),
        "status": "committed",
        "iteration": iter_num
    }

    coll_write = db.get_collection(COLL_NAME, write_concern=write_concern)
    WRITE_LISTENER.last_write_node = "unknown"

    # 1. WRITE
    t_write_start = time.perf_counter()
    write_err = None
    write_acknowledged = False
    try:
        with cluster_client.start_session(causal_consistency=False) as session:
            coll_write.insert_one(payload, session=session)
            write_acknowledged = True
    except Exception as e:
        write_err = type(e).__name__
        write_acknowledged = False
    t_write_end = time.perf_counter()
    write_ms = (t_write_end - t_write_start) * 1000

    write_node = WRITE_LISTENER.last_write_node
    if write_node == "unknown" and cluster_client.primary:
        write_node = f"{cluster_client.primary[0]}:{cluster_client.primary[1]}"

    # Initialize read results
    pri_res = {"outcome": "SKIPPED", "latency_ms": 0.0, "serving_node": "N/A", "err_detail": None}
    sec_res = {"outcome": "SKIPPED", "latency_ms": 0.0, "serving_node": "N/A", "err_detail": None}

    # 2 & 3. READ PRIMARY & SECONDARY (if write acknowledged)
    if write_acknowledged:
        coll_read = db.get_collection(COLL_NAME, read_concern=read_concern)
        with cluster_client.start_session(causal_consistency=False) as session:
            # Primary read
            pri_res = execute_read_with_node(coll_read, {"_id": doc_id}, session, ReadPreference.PRIMARY)
            # Secondary read
            sec_res = execute_read_with_node(coll_read, {"_id": doc_id}, session, ReadPreference.SECONDARY)

    write_status = "ACK" if write_acknowledged else f"FAILED({write_err})"
    pri_label = pri_res["outcome"] if pri_res["outcome"] != "ERROR" else f"ERROR({pri_res['err_detail']})"
    sec_label = sec_res["outcome"] if sec_res["outcome"] != "ERROR" else f"ERROR({sec_res['err_detail']})"

    print(f"\n  --- [Repetition {iter_num}/{total_iters}] ---")
    print(f"    Write:             {write_ms:7.2f} ms | Status: {write_status:<10} | Node: {write_node}")
    if write_acknowledged:
        print(f"    Primary Read:      {pri_res['latency_ms']:7.2f} ms | Outcome: {pri_label:<10} | Node: {pri_res['serving_node']}")
        print(f"    Secondary Read:    {sec_res['latency_ms']:7.2f} ms | Outcome: {sec_label:<10} | Node: {sec_res['serving_node']}")
    else:
        print(f"    Reads:             SKIPPED (Availability Failure on write)")

    return {
        "iter": iter_num,
        "write_ms": write_ms,
        "write_status": write_status,
        "write_acknowledged": write_acknowledged,
        "write_err": write_err,
        "write_node": write_node,
        "pri_outcome": pri_res["outcome"],
        "pri_latency_ms": pri_res["latency_ms"],
        "pri_node": pri_res["serving_node"],
        "sec_outcome": sec_res["outcome"],
        "sec_latency_ms": sec_res["latency_ms"],
        "sec_node": sec_res["serving_node"],
    }


def test_scenario(cluster_client, name, write_concern, read_concern, iterations=NUM_ITERATIONS):
    print(f"\n" + "=" * 95)
    print(f"Scenario: {name} ({iterations} repetitions)")
    print("=" * 95)

    # Clean collection before scenario
    cluster_client[DB_NAME][COLL_NAME].drop()

    results = []
    for i in range(1, iterations + 1):
        res = run_trial(cluster_client, write_concern, read_concern, i, iterations)
        results.append(res)

    # Aggregations
    total_writes = len(results)
    ack_writes = sum(1 for r in results if r["write_acknowledged"])
    avail_failures = total_writes - ack_writes
    avg_write = sum(r["write_ms"] for r in results) / total_writes

    ack_results = [r for r in results if r["write_acknowledged"]]
    if ack_results:
        avg_read_pri = sum(r["pri_latency_ms"] for r in ack_results) / len(ack_results)
        avg_read_sec = sum(r["sec_latency_ms"] for r in ack_results) / len(ack_results)
        pri_pass = sum(1 for r in ack_results if r["pri_outcome"] == "PASS")
        pri_viol = sum(1 for r in ack_results if r["pri_outcome"] == "VIOLATION")
        pri_err = sum(1 for r in ack_results if r["pri_outcome"] == "ERROR")
        sec_pass = sum(1 for r in ack_results if r["sec_outcome"] == "PASS")
        sec_viol = sum(1 for r in ack_results if r["sec_outcome"] == "VIOLATION")
        sec_err = sum(1 for r in ack_results if r["sec_outcome"] == "ERROR")
    else:
        avg_read_pri = avg_read_sec = 0.0
        pri_pass = pri_viol = pri_err = 0
        sec_pass = sec_viol = sec_err = 0

    print("\n" + "-" * 95)
    print(f"  Scenario Summary: {name}")
    print(f"    Writes:                    ACK={ack_writes}/{total_writes} | Availability Failures={avail_failures}")
    print(f"    Avg Write Latency:         {avg_write:8.2f} ms")
    if ack_results:
        print(f"    PRIMARY Read Latency:      {avg_read_pri:8.2f} ms | PASS={pri_pass}, VIOLATION={pri_viol}, ERROR={pri_err}")
        print(f"    SECONDARY Read Latency:    {avg_read_sec:8.2f} ms | PASS={sec_pass}, VIOLATION={sec_viol}, ERROR={sec_err}")
    else:
        print(f"    Reads:                     All skipped due to write availability failure.")
    print("-" * 95)

    return {
        "scenario": name,
        "total_writes": total_writes,
        "ack_writes": ack_writes,
        "avail_failures": avail_failures,
        "avg_write": avg_write,
        "avg_read_pri": avg_read_pri,
        "pri_counts": (pri_pass, pri_viol, pri_err),
        "avg_read_sec": avg_read_sec,
        "sec_counts": (sec_pass, sec_viol, sec_err),
        "iterations": iterations,
        "rep_details": results
    }


def main():
    print("=" * 95)
    print("  MongoDB Read-Your-Writes (RYW) Under Established Network Partition Test")
    print("  Environment: 2-3 Partition Active (Minority [mongo1, mongo2] | Majority [mongo3, mongo4, mongo5])")
    print("=" * 95)

    # 1. Verify Topology
    verify_partition_topology()

    # 2. Connect client with event listener for write node tracking
    print(f"Connecting PyMongo to Cluster URI:\n  {URI}\n")
    cluster_client = MongoClient(
        URI,
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        retryWrites=False,
        event_listeners=[WRITE_LISTENER]
    )

    scenarios = [
        ("W=majority, R=majority", WriteConcern(w="majority", wtimeout=WRITE_TIMEOUT_MS), ReadConcern("majority")),
        ("W=majority, R=local",    WriteConcern(w="majority", wtimeout=WRITE_TIMEOUT_MS), ReadConcern("local")),
        ("W=1,        R=majority", WriteConcern(w=1),                                       ReadConcern("majority")),
        ("W=1,        R=local",    WriteConcern(w=1),                                       ReadConcern("local")),
    ]

    all_summaries = []
    try:
        for name, wc, rc in scenarios:
            summary = test_scenario(cluster_client, name, wc, rc, iterations=NUM_ITERATIONS)
            all_summaries.append(summary)

        # Final comparison table
        print("\n" + "=" * 105)
        print("                 OVERALL RYW ESTABLISHED NETWORK PARTITION SUMMARY TABLE")
        print("=" * 105)
        print(f"{'Scenario':<25} | {'Write Status':<15} | {'Write Lat':<10} | {'Primary Read':<22} | {'Secondary Read':<22}")
        print("-" * 105)
        for s in all_summaries:
            w_stat = f"{s['ack_writes']}/{s['total_writes']} ACK"
            if s["avail_failures"] > 0:
                w_stat += f" ({s['avail_failures']} Fail)"
            w_lat = f"{s['avg_write']:.1f} ms"

            if s["ack_writes"] > 0:
                pri_c = s["pri_counts"]
                pri_str = f"P:{pri_c[0]} V:{pri_c[1]} E:{pri_c[2]} ({s['avg_read_pri']:.1f}ms)"
                sec_c = s["sec_counts"]
                sec_str = f"P:{sec_c[0]} V:{sec_c[1]} E:{sec_c[2]} ({s['avg_read_sec']:.1f}ms)"
            else:
                pri_str = "N/A (Write Failed)"
                sec_str = "N/A (Write Failed)"

            print(f"{s['scenario']:<25} | {w_stat:<15} | {w_lat:<10} | {pri_str:<22} | {sec_str:<22}")
        print("=" * 105)
        print("\nLegend: P=PASS (Read Your Write observed), V=VIOLATION (Write missing), E=ERROR (Read exception)")

    finally:
        try:
            cleanup_db = cluster_client[DB_NAME]
            cleanup_db[COLL_NAME].drop()
        except Exception:
            pass
        cluster_client.close()

        if HEAL_ON_EXIT:
            print("\n==> HEAL_ON_EXIT=true: Flushing iptables on all nodes...")
            import subprocess
            for c in ["mongo1", "mongo2", "mongo3", "mongo4", "mongo5"]:
                subprocess.run(["docker", "exec", "-u", "0", c, "iptables", "-F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("==> Partition healed.")
        else:
            print("\n==> Benchmark completed. Cluster remains in the established partitioned state.")
            print("    When you are ready to heal the cluster, run:")
            print("    for c in mongo1 mongo2 mongo3 mongo4 mongo5; do docker exec -u 0 $c iptables -F; done")


if __name__ == "__main__":
    main()
