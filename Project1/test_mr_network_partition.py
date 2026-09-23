#!/usr/bin/env python3
"""
MongoDB Monotonic Reads (MR) Benchmark Under a 2–3 Network Partition.

Topology:
  Minority Side: mongo1 (PRIMARY), mongo2 (SECONDARY, fixed Read #1 reader)
  Majority Side: mongo3, mongo4, mongo5 (SECONDARIES, Read #2 target pool)

Experiment Flow:
  1. Establish v1 on all 5 nodes (w="majority")
  2. Apply 500 ms replication delay from mongo1 to mongo3, mongo4, mongo5 (NOT mongo2)
  3. Write v2 with scenario write concern (w:1 or w:majority)
  4. Start single ClientSession (causal_consistency configurable)
  5. Read #1 on mongo2 (READ1_PREF) until v2 observed (~2 ms polling)
  6. Immediately apply bidirectional partition: {mongo1, mongo2} X {mongo3, mongo4, mongo5}
  7. Verify partition rules
  8. Capture diagnostic snapshot of majority replicas (mongo3, mongo4, mongo5)
  9. Read #2 with MAJORITY_SIDE_PREF in the same reader session
  10. Classify outcome: PASS (v2), VIOLATION (v1), ERROR (availability/timeout), INVALID PRECONDITION
  11. Heal partition, remove delay, and restore cluster in finally block.
"""

import os
import sys
import time
import uuid
import socket
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient, monitoring
from pymongo.read_preferences import Secondary
from pymongo.write_concern import WriteConcern
from pymongo.read_concern import ReadConcern

# Replica set connection URI inside the docker network
URI = os.getenv(
    "MONGODB_URI",
    "mongodb://mongo1:27017,mongo2:27017,mongo3:27017,mongo4:27017,mongo5:27017/?replicaSet=rs0"
)

DB_NAME = "mr_partition_test_db"
COLL_NAME = "records"
NUM_ITERATIONS = 15
CAUSAL_CONSISTENCY = True  # Set to True to test with causal consistency enabled
PRIMARY_CONTAINER = "mongo1"

MINORITY_NODES = ["mongo1", "mongo2"]
MAJORITY_NODES = ["mongo3", "mongo4", "mongo5"]
ALL_NODES = MINORITY_NODES + MAJORITY_NODES

# PyMongo Read Preferences targeting specific partition sides / roles
READ1_PREF = Secondary(tag_sets=[{"reader": "first"}])
MAJORITY_SIDE_PREF = Secondary(tag_sets=[{"partition": "majority"}])


class ServerTrackingListener(monitoring.CommandListener):
    """Monitors read commands to identify which server node handled each query."""
    def __init__(self):
        self.last_find_node = "unknown"

    def started(self, event):
        pass

    def succeeded(self, event):
        if event.command_name == "find":
            addr = getattr(event, "address", None)
            conn = getattr(event, "connection_id", None)
            if addr and isinstance(addr, (list, tuple)):
                self.last_find_node = f"{addr[0]}:{addr[1]}"
            elif conn and isinstance(conn, (list, tuple)):
                self.last_find_node = f"{conn[0]}:{conn[1]}"
            elif addr:
                self.last_find_node = str(addr)
            elif conn:
                self.last_find_node = str(conn)

    def failed(self, event):
        addr = getattr(event, "address", None)
        conn = getattr(event, "connection_id", None)
        if addr and isinstance(addr, (list, tuple)):
            self.last_find_node = f"{addr[0]}:{addr[1]}"
        elif conn and isinstance(conn, (list, tuple)):
            self.last_find_node = f"{conn[0]}:{conn[1]}"


TRACKING_LISTENER = ServerTrackingListener()


def get_docker_client():
    """Initializes docker client if available."""
    try:
        import docker
        return docker.from_env()
    except Exception:
        return None


DOCKER_CLIENT = get_docker_client()


def exec_container_command(container_name, cmd):
    """Executes a command inside a container as root."""
    if DOCKER_CLIENT:
        try:
            container = DOCKER_CLIENT.containers.get(container_name)
            res = container.exec_run(cmd, user="0")
            return res.exit_code, res.output.decode("utf-8", errors="replace")
        except Exception:
            pass
    # Fallback to subprocess CLI
    try:
        if isinstance(cmd, list):
            cmd_args = ["docker", "exec", "-u", "0", container_name] + cmd
        else:
            cmd_args = ["docker", "exec", "-u", "0", container_name, "sh", "-c", cmd]
        p = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        return p.returncode, p.stdout + p.stderr
    except Exception as e:
        return 1, str(e)


def get_node_ip_map():
    """Resolves IP addresses for all cluster nodes."""
    ip_map = {}
    for node in ALL_NODES:
        ip = None
        if DOCKER_CLIENT:
            try:
                c = DOCKER_CLIENT.containers.get(node)
                networks = c.attrs.get("NetworkSettings", {}).get("Networks", {})
                for net_info in networks.values():
                    if net_info.get("IPAddress"):
                        ip = net_info["IPAddress"]
                        break
            except Exception:
                pass
        if not ip:
            try:
                ip = socket.gethostbyname(node)
            except Exception:
                pass
        if not ip:
            code, out = exec_container_command(node, "hostname -i")
            if code == 0 and out.strip():
                ip = out.strip().split()[0]
        if not ip:
            raise RuntimeError(f"Could not resolve IP address for node {node}")
        ip_map[node] = ip
    return ip_map


IP_MAP = get_node_ip_map()


def apply_replication_delay(delay_ms=500):
    """
    Installs tc netem delay on mongo1 towards mongo3, mongo4, and mongo5,
    leaving traffic towards mongo2 unaffected.
    """
    m3_ip = IP_MAP["mongo3"]
    m4_ip = IP_MAP["mongo4"]
    m5_ip = IP_MAP["mongo5"]

    tc_cmd = (
        f"tc qdisc del dev eth0 root 2>/dev/null || true; "
        f"tc qdisc add dev eth0 root handle 1: prio && "
        f"tc qdisc add dev eth0 parent 1:3 handle 30: netem delay {delay_ms}ms && "
        f"tc filter add dev eth0 protocol ip parent 1:0 u32 match ip dst {m3_ip} flowid 1:3 && "
        f"tc filter add dev eth0 protocol ip parent 1:0 u32 match ip dst {m4_ip} flowid 1:3 && "
        f"tc filter add dev eth0 protocol ip parent 1:0 u32 match ip dst {m5_ip} flowid 1:3"
    )

    code, out = exec_container_command(PRIMARY_CONTAINER, ["sh", "-c", tc_cmd])
    if code != 0:
        # Fallback to nicolaka/netshoot if needed
        netshoot_cmd = f"docker run --rm --cap-add=NET_ADMIN --net=container:{PRIMARY_CONTAINER} nicolaka/netshoot sh -c '{tc_cmd}'"
        subprocess.run(netshoot_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def remove_replication_delay():
    """Removes tc qdisc rules on mongo1."""
    exec_container_command(PRIMARY_CONTAINER, "tc qdisc del dev eth0 root 2>/dev/null || true")


def init_partition_chains():
    """
    Ensures a dedicated MR_PARTITION chain exists on all containers and is hooked
    at the top of INPUT and OUTPUT. This avoids flushing or disturbing Docker's default rules.
    """
    setup_script = (
        "iptables -N MR_PARTITION 2>/dev/null || true; "
        "iptables -C INPUT -j MR_PARTITION 2>/dev/null || iptables -I INPUT 1 -j MR_PARTITION; "
        "iptables -C OUTPUT -j MR_PARTITION 2>/dev/null || iptables -I OUTPUT 1 -j MR_PARTITION; "
        "iptables -F MR_PARTITION"
    )
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(lambda n: exec_container_command(n, ["sh", "-c", setup_script]), ALL_NODES))


def create_partition():
    """
    Populates the dedicated MR_PARTITION chain on each node with DROP rules:
      Minority: {mongo1, mongo2} drops traffic to/from majority {mongo3, mongo4, mongo5}
      Majority: {mongo3, mongo4, mongo5} drops traffic to/from minority {mongo1, mongo2}
    Executed in parallel across all 5 containers for sub-millisecond setup.
    """
    minority_ips = [IP_MAP[n] for n in MINORITY_NODES]
    majority_ips = [IP_MAP[n] for n in MAJORITY_NODES]

    # Minority nodes block traffic to/from majority
    min_cmds = ["iptables -F MR_PARTITION"]
    for ip in majority_ips:
        min_cmds.append(f"iptables -A MR_PARTITION -s {ip} -j DROP")
        min_cmds.append(f"iptables -A MR_PARTITION -d {ip} -j DROP")
    min_script = " && ".join(min_cmds)

    # Majority nodes block traffic to/from minority
    maj_cmds = ["iptables -F MR_PARTITION"]
    for ip in minority_ips:
        maj_cmds.append(f"iptables -A MR_PARTITION -s {ip} -j DROP")
        maj_cmds.append(f"iptables -A MR_PARTITION -d {ip} -j DROP")
    maj_script = " && ".join(maj_cmds)

    def apply_node_partition(node):
        script = min_script if node in MINORITY_NODES else maj_script
        exec_container_command(node, ["sh", "-c", script])

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(apply_node_partition, ALL_NODES))


def heal_partition():
    """Flushes only the MR_PARTITION chain on all 5 nodes, preserving Docker's default rules."""
    def flush_node(node):
        exec_container_command(node, "iptables -F MR_PARTITION 2>/dev/null || true")

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(flush_node, ALL_NODES))


def verify_partition():
    """
    Comprehensively verifies that all cross-partition edges are blocked
    and intra-partition connectivity is preserved across all nodes.
    Executed concurrently across all check pairs.
    """
    checks = []

    # 1. Intra-partition checks (MUST succeed: expect_success=True)
    # Minority: mongo1 <-> mongo2
    checks.append(("mongo1", f"ping -c 1 -W 0.25 {IP_MAP['mongo2']}", True))
    # Majority: mongo3 <-> mongo4, mongo3 <-> mongo5, mongo4 <-> mongo5
    checks.append(("mongo3", f"ping -c 1 -W 0.25 {IP_MAP['mongo4']}", True))
    checks.append(("mongo3", f"ping -c 1 -W 0.25 {IP_MAP['mongo5']}", True))
    checks.append(("mongo4", f"ping -c 1 -W 0.25 {IP_MAP['mongo5']}", True))

    # 2. Cross-partition checks (MUST fail/drop: expect_success=False)
    # Minority -> Majority
    for min_node in MINORITY_NODES:
        for maj_node in MAJORITY_NODES:
            checks.append((min_node, f"ping -c 1 -W 0.25 {IP_MAP[maj_node]}", False))
    # Majority -> Minority
    for maj_node in MAJORITY_NODES:
        for min_node in MINORITY_NODES:
            checks.append((maj_node, f"ping -c 1 -W 0.25 {IP_MAP[min_node]}", False))

    def run_check(item):
        node, cmd, expect_success = item
        code, _ = exec_container_command(node, cmd)
        return (code == 0) if expect_success else (code != 0)

    # Run all 16 checks concurrently
    with ThreadPoolExecutor(max_workers=len(checks)) as executor:
        results = list(executor.map(run_check, checks))

    return all(results)


def capture_majority_snapshot(doc_id):
    """
    Captures diagnostic state and document version directly from mongo3, mongo4, mongo5 concurrently.
    Returns a dict mapping node -> {"state": stateStr, "v": version}.
    """
    def probe_node(node):
        state = "UNKNOWN"
        ver = None
        try:
            with MongoClient(f"mongodb://{node}:27017/?directConnection=true", serverSelectionTimeoutMS=1000) as direct_client:
                hello = direct_client.admin.command("hello")
                if hello.get("isWritablePrimary"):
                    state = "PRIMARY"
                elif hello.get("secondary"):
                    state = "SECONDARY"
                else:
                    state = "OTHER"

                doc = direct_client[DB_NAME][COLL_NAME].find_one({"_id": doc_id})
                if doc:
                    ver = doc.get("v")
        except Exception as e:
            state = f"ERR({type(e).__name__})"
        return node, {"state": state, "v": ver}

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = dict(executor.map(probe_node, MAJORITY_NODES))
    return results


def ensure_mongo1_primary(cluster_client, timeout=60):
    """
    Ensures mongo1 is running and has reclaimed PRIMARY status.
    Uses priority-assisted step-down if another node temporarily holds PRIMARY.
    """
    start_time = time.time()
    last_stepdown_time = 0

    while time.time() - start_time < timeout:
        status = None
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
                    now = time.time()
                    if now - last_stepdown_time > 4:
                        curr_pri = next((m["name"] for m in status["members"] if m.get("stateStr") == "PRIMARY"), None)
                        if curr_pri and PRIMARY_CONTAINER not in curr_pri:
                            try:
                                with MongoClient(f"mongodb://{curr_pri}/?directConnection=true", serverSelectionTimeoutMS=2000) as temp_client:
                                    temp_client.admin.command("replSetStepDown", 15)
                                    last_stepdown_time = now
                            except Exception:
                                pass
        time.sleep(1)

    raise RuntimeError(f"Timed out after {timeout}s waiting for {PRIMARY_CONTAINER} to become PRIMARY!")


def wait_cluster_synchronized(cluster_client, timeout=30):
    """Waits until all 5 nodes are healthy and synchronized."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            status = cluster_client.admin.command("replSetGetStatus")
            members = status.get("members", [])
            if len(members) == 5:
                all_healthy = all(
                    m.get("health") == 1 and m.get("stateStr") in ("PRIMARY", "SECONDARY")
                    for m in members
                )
                if all_healthy:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def restore_cluster(cluster_client):
    """Full self-healing and restoration of the cluster to standard baseline."""
    heal_partition()
    remove_replication_delay()
    init_partition_chains()
    ensure_mongo1_primary(cluster_client)
    wait_cluster_synchronized(cluster_client)


def verify_all_nodes_have_version(doc_id, expected_version, timeout=15):
    """Directly verifies that all 5 nodes observe the expected document version."""
    start = time.time()
    while time.time() - start < timeout:
        all_match = True
        for node in ALL_NODES:
            try:
                with MongoClient(f"mongodb://{node}:27017/?directConnection=true", serverSelectionTimeoutMS=2000) as direct_client:
                    doc = direct_client[DB_NAME][COLL_NAME].find_one({"_id": doc_id})
                    if not doc or doc.get("v") != expected_version:
                        all_match = False
                        break
            except Exception:
                all_match = False
                break
        if all_match:
            return True
        time.sleep(0.05)
    return False


def cleanup_db(client):
    """Drops the test database."""
    try:
        client.drop_database(DB_NAME)
    except Exception:
        pass


def run_iteration(writer_client, reader_client, cluster_client, write_concern, read_concern, causal_consistency, iter_num, total_iters):
    """Executes a single repetition of the MR network partition experiment."""
    doc_id = str(uuid.uuid4())
    writer_db = writer_client[DB_NAME]
    reader_db = reader_client[DB_NAME]

    # Pre-flight baseline: ensure mongo1 is PRIMARY, delay removed, and healed
    restore_cluster(cluster_client)

    coll_write_init = writer_db.get_collection(COLL_NAME, write_concern=WriteConcern(w="majority", wtimeout=10000))
    coll_write_v2 = writer_db.get_collection(COLL_NAME, write_concern=write_concern)

    try:
        # Step 1: Write v1 with w="majority" and verify on all 5 nodes
        coll_write_init.insert_one({"_id": doc_id, "v": 1, "created_at": time.time()})
        if not verify_all_nodes_have_version(doc_id, expected_version=1, timeout=15):
            return {
                "iter": iter_num,
                "write_v2_ms": 0.0,
                "fast_detect_ms": 0.0,
                "part_apply_ms": 0.0,
                "read2_latency_ms": 0.0,
                "outcome": "INVALID PRECONDITION",
                "err_detail": "Initial v1 was not confirmed on all 5 nodes",
                "read2_server": "N/A",
                "observed_v": None,
                "snapshot": {},
                "recov_sec": 0.0
            }

        # Step 2: Apply 500 ms replication delay from mongo1 to mongo3, mongo4, mongo5
        apply_replication_delay(delay_ms=500)

        # Step 3: Write v2 on mongo1
        t_write_start = time.perf_counter()
        try:
            coll_write_v2.update_one({"_id": doc_id}, {"$set": {"v": 2, "updated_at": time.time()}})
            t_write_end = time.perf_counter()
            write_v2_ms = (t_write_end - t_write_start) * 1000
        except Exception as e:
            return {
                "iter": iter_num,
                "write_v2_ms": 0.0,
                "fast_detect_ms": 0.0,
                "part_apply_ms": 0.0,
                "read2_latency_ms": 0.0,
                "outcome": "INVALID PRECONDITION",
                "err_detail": f"v2 write failed: {type(e).__name__}",
                "read2_server": "N/A",
                "observed_v": None,
                "snapshot": {},
                "recov_sec": 0.0
            }

        # Step 4 & 5: Single Reader Session -> Read #1 on mongo2 until v2 observed
        coll_read1 = reader_db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=READ1_PREF)
        coll_read2 = reader_db.get_collection(COLL_NAME, read_concern=read_concern, read_preference=MAJORITY_SIDE_PREF)

        with reader_client.start_session(causal_consistency=causal_consistency) as reader_session:
            t_poll_start = time.perf_counter()
            fast_detected_v2 = False
            poll_deadline = time.time() + 5.0

            while time.time() < poll_deadline:
                doc1 = coll_read1.find_one({"_id": doc_id}, session=reader_session)
                if doc1 and doc1.get("v") == 2:
                    fast_detected_v2 = True
                    break
                time.sleep(0.002)

            t_read1_v2 = time.perf_counter()
            fast_detect_ms = (t_read1_v2 - t_poll_start) * 1000

            if not fast_detected_v2:
                return {
                    "iter": iter_num,
                    "write_v2_ms": write_v2_ms,
                    "fast_detect_ms": fast_detect_ms,
                    "part_apply_ms": 0.0,
                    "read2_latency_ms": 0.0,
                    "outcome": "INVALID PRECONDITION",
                    "err_detail": "mongo2 timed out waiting to observe v2",
                    "read2_server": "N/A",
                    "observed_v": None,
                    "snapshot": {},
                    "recov_sec": 0.0
                }

            # Step 6: Immediately create bidirectional partition {mongo1, mongo2} X {mongo3, mongo4, mongo5}
            t_part_start = time.perf_counter()
            create_partition()
            t_part_end = time.perf_counter()
            part_apply_ms = (t_part_end - t_part_start) * 1000

            # Step 7: READ #2 IMMEDIATELY on Majority Side (critical measurement)
            TRACKING_LISTENER.last_find_node = "unknown"
            t_r2_start = time.perf_counter()
            outcome = "ERROR"
            err_detail = None
            observed_v = None
            read2_server = "unknown"

            try:
                cursor = coll_read2.find({"_id": doc_id}, session=reader_session).limit(1)
                doc2 = next(cursor, None)
                if cursor.address:
                    read2_server = f"{cursor.address[0]}:{cursor.address[1]}"
                elif TRACKING_LISTENER.last_find_node != "unknown":
                    read2_server = TRACKING_LISTENER.last_find_node

                observed_v = doc2.get("v") if doc2 else None
                if observed_v == 2:
                    outcome = "PASS"
                elif observed_v == 1:
                    outcome = "VIOLATION"
                else:
                    outcome = "ERROR"
                    err_detail = f"Unexpected version: {observed_v}"
            except Exception as e:
                outcome = "ERROR"
                err_detail = type(e).__name__
                if TRACKING_LISTENER.last_find_node != "unknown":
                    read2_server = TRACKING_LISTENER.last_find_node

            t_r2_end = time.perf_counter()
            read2_latency_ms = (t_r2_end - t_r2_start) * 1000

            # Step 8: Verify partition rules (post-read precondition check)
            if not verify_partition():
                outcome = "INVALID PRECONDITION"
                err_detail = "Partition verification failed (isolation rules not strictly effective)"

            # Step 9: Diagnostic snapshot of majority replicas (post-read diagnostic state)
            snapshot = capture_majority_snapshot(doc_id)

    finally:
        # Step 10: Heal partition & restore cluster in finally block
        t_recov_start = time.perf_counter()
        restore_cluster(cluster_client)
        t_recov_end = time.perf_counter()
        recov_sec = t_recov_end - t_recov_start

    # Format snapshot for concise logging
    snap_str = " ".join([f"{n}:v={s.get('v')}" for n, s in sorted(snapshot.items())])
    out_label = outcome if outcome != "ERROR" else f"ERROR({err_detail})"

    print(f"\n  --- [Repetition {iter_num}/{total_iters}] ---")
    print(f"    Write v2 Latency:     {write_v2_ms:7.2f} ms")
    print(f"    Fast Detect v2 (m2):  {fast_detect_ms:7.2f} ms")
    print(f"    Partition Applied In: {part_apply_ms:7.2f} ms")
    print(f"    Read #2 Served By:    {read2_server} in {read2_latency_ms:7.2f} ms | Outcome: {out_label} (Observed v={observed_v})")
    print(f"    Majority Snapshot:    {snap_str} (captured post-Read #2)")
    print(f"    Cluster Recovery:     Restored baseline in {recov_sec:.1f}s")

    return {
        "iter": iter_num,
        "write_v2_ms": write_v2_ms,
        "fast_detect_ms": fast_detect_ms,
        "part_apply_ms": part_apply_ms,
        "read2_latency_ms": read2_latency_ms,
        "outcome": outcome,
        "err_detail": err_detail,
        "read2_server": read2_server,
        "observed_v": observed_v,
        "snapshot": snapshot,
        "recov_sec": recov_sec
    }


def test_scenario(writer_client, reader_client, cluster_client, name, write_concern, read_concern, causal_consistency, iterations=NUM_ITERATIONS):
    """Runs all repetitions for a specific write concern and read concern scenario."""
    print(f"\n" + "=" * 95)
    print(f"Scenario: {name} (Causal Consistency: {causal_consistency}, {iterations} repetitions)")
    print("=" * 95)

    writer_client[DB_NAME][COLL_NAME].drop()

    results = []
    for i in range(1, iterations + 1):
        res = run_iteration(
            writer_client=writer_client,
            reader_client=reader_client,
            cluster_client=cluster_client,
            write_concern=write_concern,
            read_concern=read_concern,
            causal_consistency=causal_consistency,
            iter_num=i,
            total_iters=iterations
        )
        results.append(res)

    valid_mr_comparisons = [r for r in results if r["outcome"] in ("PASS", "VIOLATION")]
    passes = sum(1 for r in results if r["outcome"] == "PASS")
    violations = sum(1 for r in results if r["outcome"] == "VIOLATION")
    errors = sum(1 for r in results if r["outcome"] == "ERROR")
    invalid_preconditions = sum(1 for r in results if r["outcome"] == "INVALID PRECONDITION")

    valid_count = len(valid_mr_comparisons)
    pass_rate = (passes / valid_count * 100) if valid_count > 0 else 0.0

    avg_read2 = (sum(r["read2_latency_ms"] for r in valid_mr_comparisons) / valid_count) if valid_count > 0 else 0.0
    valid_detect_trials = [r for r in results if r["outcome"] in ("PASS", "VIOLATION", "ERROR")]
    avg_fast_detect = (sum(r["fast_detect_ms"] for r in valid_detect_trials) / len(valid_detect_trials)) if valid_detect_trials else 0.0

    print("\n" + "-" * 95)
    print(f"  Scenario Summary: {name} (Causal Consistency={causal_consistency})")
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
    print(f"    Avg Read #2 Latency:        {avg_read2:8.2f} ms (valid comparisons only)")
    print("-" * 95)

    return {
        "scenario": name,
        "causal_consistency": causal_consistency,
        "trials": len(results),
        "valid_count": valid_count,
        "passes": passes,
        "violations": violations,
        "errors": errors,
        "invalid_preconditions": invalid_preconditions,
        "pass_rate": pass_rate,
        "avg_fast_detect": avg_fast_detect,
        "avg_read2": avg_read2,
        "rep_details": results
    }


def print_summary_table(all_results, title="MONOTONIC READS UNDER NETWORK PARTITION BENCHMARK SUMMARY"):
    """Prints formatted ASCII summary table of benchmark results."""
    print("\n" + "=" * 125)
    print(f"                    {title}")
    print("=" * 125)
    header = (
        f"{'Scenario':<28} | {'CC':<5} | {'Trials':<6} | {'Valid':<5} | {'Pass':<5} | "
        f"{'Viol':<5} | {'MR Pass Rate':<12} | {'Error':<5} | {'Invalid':<7} | {'Avg Fast':<10} | {'Avg Read #2':<11}"
    )
    print(header)
    print("-" * 125)
    for r in all_results:
        rate_str = f"{r['pass_rate']:5.1f}%" if r["valid_count"] > 0 else "N/A"
        cc_str = "True" if r["causal_consistency"] else "False"
        line = (
            f"{r['scenario']:<28} | "
            f"{cc_str:<5} | "
            f"{r['trials']:>6} | "
            f"{r['valid_count']:>5} | "
            f"{r['passes']:>5} | "
            f"{r['violations']:>5} | "
            f"{rate_str:>12} | "
            f"{r['errors']:>5} | "
            f"{r['invalid_preconditions']:>7} | "
            f"{r['avg_fast_detect']:>7.2f} ms | "
            f"{r['avg_read2']:>8.2f} ms"
        )
        print(line)
    print("=" * 125)
    print("\nOutcome Legend & Methodology:")
    print("  Valid MR Comparisons : Pass + Violation (trials where Read #1 detected v2, partition created, and Read #2 returned data).")
    print("  Pass                 : Read #2 observed v2 (Monotonic Reads preserved across partition).")
    print("  Violation            : Read #2 observed v1 (Monotonic Reads violated - read went backwards).")
    print("  Error                : Read #2 failed/timed out (availability failure during partition).")
    print("  Invalid Precondition : Precondition failed (v2 write failed, mongo2 didn't see v2, or partition verify failed).")
    print("  MR Pass Rate         : Pass / Valid MR comparisons = Pass / (Pass + Violation).")
    print("=" * 125)


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else NUM_ITERATIONS

    print("=" * 95)
    print("MongoDB Monotonic Reads (MR) Benchmark Under 2–3 Network Partition")
    print("=" * 95)
    print(f"Iterations per scenario: {iterations}")
    print(f"Causal Consistency:      {CAUSAL_CONSISTENCY}")
    print("Connecting to MongoDB Replica Set...")

    cluster_client = MongoClient(URI, serverSelectionTimeoutMS=30000, heartbeatFrequencyMS=500)
    writer_client = MongoClient(URI, serverSelectionTimeoutMS=30000, heartbeatFrequencyMS=500)
    reader_client = MongoClient(
        URI,
        serverSelectionTimeoutMS=30000,
        heartbeatFrequencyMS=500,
        event_listeners=[TRACKING_LISTENER]
    )

    cluster_client.admin.command('ping')
    writer_client.admin.command('ping')
    reader_client.admin.command('ping')
    print("Connected clients successfully!")

    # Initial cluster restoration
    print(f"Ensuring initial state: restoring baseline cluster...")
    restore_cluster(cluster_client)
    cleanup_db(writer_client)
    print("Baseline verified and cluster clean.")

    test_cases = [
        {
            "name": "Case 1: W=majority, R=majority",
            "wc": WriteConcern(w="majority", wtimeout=10000),
            "rc": ReadConcern("majority")
        },
        # {
        #     "name": "Case 2: W=1,        R=majority",
        #     "wc": WriteConcern(w=1),
        #     "rc": ReadConcern("majority")
        # },
        # {
        #     "name": "Case 3: W=1,        R=local",
        #     "wc": WriteConcern(w=1),
        #     "rc": ReadConcern("local")
        # },
        # {
        #     "name": "Case 4: W=majority, R=local",
        #     "wc": WriteConcern(w="majority", wtimeout=10000),
        #     "rc": ReadConcern("local")
        # }
    ]

    all_results = []
    try:
        for tc in test_cases:
            res = test_scenario(
                writer_client=writer_client,
                reader_client=reader_client,
                cluster_client=cluster_client,
                name=tc["name"],
                write_concern=tc["wc"],
                read_concern=tc["rc"],
                causal_consistency=CAUSAL_CONSISTENCY,
                iterations=iterations
            )
            all_results.append(res)

        print_summary_table(all_results)

    finally:
        cleanup_db(writer_client)
        restore_cluster(cluster_client)
        cluster_client.close()
        writer_client.close()
        reader_client.close()
        print("\nCleaned up all test databases and verified cluster baseline. Benchmark complete.\n")


if __name__ == "__main__":
    main()
