$ErrorActionPreference = "Stop"

$ClientContainer = "wfr-normal-client"
$ClusterNetwork = "mongoCluster"
$Nodes = @("mongo1", "mongo2", "mongo3", "mongo4", "mongo5")

foreach ($node in $Nodes) {
    $running = docker ps --filter "name=^/$node$" --format "{{.Names}}"
    if ($running -notcontains $node) {
        throw "$node is not running. Run 'docker compose up -d' first."
    }
}

$existing = docker ps -a --filter "name=^/$ClientContainer$" --format "{{.Names}}"
if ($existing -contains $ClientContainer) {
    docker rm -f $ClientContainer | Out-Null
}

try {
    docker run --name $ClientContainer `
        --network $ClusterNetwork `
        -v "${PWD}:/app" `
        -w /app `
        python:3.11-slim `
        bash -c "pip install --quiet pymongo && python -u test_wfr_normal.py"

    if ($LASTEXITCODE -ne 0) {
        throw "Normal-operation test exited with code $LASTEXITCODE"
    }
}
finally {
    $existing = docker ps -a --filter "name=^/$ClientContainer$" --format "{{.Names}}"
    if ($existing -contains $ClientContainer) {
        docker rm -f $ClientContainer | Out-Null
    }
}

Write-Host "Normal-operation experiment completed."
