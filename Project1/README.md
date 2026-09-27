# MongoDB Client-Centric Consistency Experiments

> **Note on Experiment Execution:**  
> Since the experiment scripts were divided between group members, the exact execution mechanisms differ:
> - **Read-Your-Writes (RYW) & Monotonic Reads (MR):** Run in an isolated container on the Docker network (`mongoCluster`) using Bash/Docker commands (compatible with macOS, Linux, and Windows).
> - **Monotonic Writes (MW) & Writes-Follow-Reads (WFR):** Run on the host system via Windows PowerShell scripts (`.ps1`).

---

## 1. MongoDB Replica Set Initialization (Common Setup)

Before running any experiments, spin up and initialize the 5-node replica set (`mongo1` through `mongo5`):

### Step 1: Start Docker Containers
```bash
docker compose up -d
```
*(Wait 5–10 seconds to allow all MongoDB instances to boot up completely.)*

### Step 2: Initialize Replica Set (`rs0`)
Configure all 5 members via `mongosh` inside the `mongo1` container:
```bash
docker exec -it mongo1 mongosh --eval 'rs.initiate({
  _id: "rs0",
  members: [
    { _id: 0, host: "mongo1:27017" },
    { _id: 1, host: "mongo2:27017" },
    { _id: 2, host: "mongo3:27017" },
    { _id: 3, host: "mongo4:27017" },
    { _id: 4, host: "mongo5:27017" }
  ]
})'
```

### Step 3: Ensure `mongo1` is PRIMARY (Priority Configuration)
Set `mongo1` to a higher priority (priority: 2) so the cluster consistently elects it as the initial PRIMARY:
```bash
docker exec -it mongo1 mongosh "mongodb://mongo1:27017/?replicaSet=rs0" --eval '
cfg = rs.conf();
cfg.members.forEach(m => {
  m.priority = m.host.includes("mongo1") ? 2 : 1;
});
rs.reconfig(cfg);'
```

### Step 4: Verify Cluster Health
Verify that `mongo1` is `PRIMARY` and all other nodes (`mongo2`–`mongo5`) are `SECONDARY`:
```bash
docker exec -it mongo1 mongosh --eval 'rs.status().members.map(m => ({ name: m.name, stateStr: m.stateStr }))'
```

---

## 2. Read-Your-Writes (RYW) & Monotonic Reads (MR) Experiments
*(Executed via Docker container on the `mongoCluster` network)*

### A. Normal Operation (Baseline Tests)

- **Read-Your-Writes (RYW):**
```bash
docker run --rm -it \
  --network mongoCluster \
  -v "$PWD":/app \
  -w /app \
  python:3.11-slim \
  bash -c "pip install pymongo && python test_ryw.py"
```

- **Monotonic Reads (MR) - Multi-Client:**
```bash
docker run --rm -it \
  --network mongoCluster \
  -v "$PWD":/app \
  -w /app \
  python:3.11-slim \
  bash -c "pip install pymongo && python test_mr.py"
```

- **Monotonic Reads (MR) - Single Client / Causal Consistency:**
```bash
docker run --rm -it \
  --network mongoCluster \
  -v "$PWD":/app \
  -w /app \
  python:3.11-slim \
  bash -c "pip install pymongo && python test_mr_single_client.py"
```

### B. Primary Failure Experiments
*Mounts `/var/run/docker.sock` to enable the benchmark runner to abruptly terminate and recover the primary container (`mongo1`).*

- **RYW Under Primary Failover:**
```bash
docker run --rm -it \
  --network mongoCluster \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD":/app \
  -w /app \
  python:3.11-slim \
  bash -c "pip install pymongo docker && python test_ryw_primary_failover.py"
```

- **MR Under Primary Failure:**
```bash
docker run --rm -it \
  --network mongoCluster \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD":/app \
  -w /app \
  python:3.11-slim \
  bash -c "pip install pymongo docker && python test_mr_primary_failure.py"
```

### C. Established Network Partition Experiments (2–3 Partition)
*Simulates a partition between Minority (`mongo1`, `mongo2`) and Majority (`mongo3`, `mongo4`, `mongo5`).*

1. **Apply Network Partition (Before running tests):**
   Follow [setup_ryw_partition.txt](setup_ryw_partition.txt) to configure `iptables` dropping cross-partition traffic.
2. **Run Tests:**
   - **RYW Under Network Partition:**
   ```bash
   docker run --rm -it \
     --network mongoCluster \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v "$PWD":/app \
     -w /app \
     python:3.11-slim \
     bash -c "pip install pymongo docker && python test_ryw_network_partition.py"
   ```
   - **MR Under Network Partition:**
   ```bash
   docker run --rm -it \
     --network mongoCluster \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v "$PWD":/app \
     -w /app \
     python:3.11-slim \
     bash -c "pip install pymongo docker && python test_mr_network_partition.py"
   ```
3. **Heal Network Partition:**
   ```bash
   for c in mongo1 mongo2 mongo3 mongo4 mongo5; do docker exec -u 0 $c iptables -F; done
   ```

---

## 3. Monotonic Writes (MW) & Writes-Follow-Reads (WFR) Experiments
*(Executed via Windows PowerShell runners)*

Run commands from the project root in Windows PowerShell after the five-node replica set is running.

### Normal Operation
```powershell
.\run_mw_normal.ps1
.\run_wfr_normal.ps1
```

### Primary Failure
```powershell
.\run_mw_failure.ps1
.\run_wfr_failure.ps1
```

### Network Partition
Each network-partition case has its own runner. This prevents `NUM_ITERATIONS = 5` from running all 25 case-iterations in one command.

```powershell
.\run_mw_np1.ps1
.\run_mw_np2.ps1
.\run_mw_np3.ps1
.\run_mw_np4.ps1
.\run_mw_np5.ps1
.\run_mw_np6.ps1
.\run_mw_np7.ps1

.\run_wfr_np1.ps1
.\run_wfr_np2.ps1
.\run_wfr_np3.ps1
.\run_wfr_np4.ps1
.\run_wfr_np5.ps1
.\run_wfr_np6.ps1
.\run_wfr_np7.ps1
```

Change `NUM_ITERATIONS` near the top of the applicable Python file. The partition runners set `SELECTED_CASE` automatically.

All summary tables begin with `Case`, followed by `Configuration`.

---

## Helpful Maintenance Commands
```bash
# Check running containers:
docker compose ps

# View container logs:
docker compose logs -f

# Stop containers:
docker compose stop

# Restart stopped containers:
docker compose start

# Teardown / Complete Reset (deletes containers & volumes):
docker compose down -v
```
