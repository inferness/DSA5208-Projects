$ErrorActionPreference = "Stop"

$ClientContainer = "mw-failure-client"
$ClusterNetwork = "mongoCluster"
$CoordDirectory = Join-Path $PWD "mw_coord"
$MaximumWaitSeconds = 300
$Nodes = @("mongo1", "mongo2", "mongo3", "mongo4", "mongo5")

function Test-ContainerExists {
    param([string]$Name)
    $found = docker ps -a --filter "name=^/$Name$" --format "{{.Names}}"
    return ($found -contains $Name)
}

function Test-ContainerRunning {
    param([string]$Name)
    $found = docker ps --filter "name=^/$Name$" --format "{{.Names}}"
    return ($found -contains $Name)
}

function Test-ConnectedToNetwork {
    param([string]$Container, [string]$Network)
    $networks = docker inspect `
        --format "{{range `$key, `$value := .NetworkSettings.Networks}}{{println `$key}}{{end}}" `
        $Container
    return ($networks -contains $Network)
}

function Connect-Safely {
    param([string]$Container, [string]$Network)
    if (-not (Test-ConnectedToNetwork $Container $Network)) {
        docker network connect --alias $Container $Network $Container
        if ($LASTEXITCODE -ne 0) { throw "Could not connect $Container to $Network" }
    }
}

function Wait-ForFile {
    param([string]$Path, [int]$TimeoutSeconds = 300)
    $started = Get-Date
    while (-not (Test-Path $Path)) {
        if (((Get-Date) - $started).TotalSeconds -ge $TimeoutSeconds) {
            throw "Timed out waiting for $Path"
        }
        if ((Test-ContainerExists $ClientContainer) -and
            -not (Test-ContainerRunning $ClientContainer)) {
            docker logs $ClientContainer
            throw "Python exited before creating $Path"
        }
        Start-Sleep -Milliseconds 200
    }
}

function New-Signal {
    param([string]$Path)
    Set-Content -Path $Path -Value "ready" -Encoding ASCII
}

foreach ($node in $Nodes) {
    if (-not (Test-ContainerExists $node)) {
        throw "$node does not exist. Run 'docker compose up -d' first."
    }
    if (-not (Test-ContainerRunning $node)) { docker start $node | Out-Null }
    Connect-Safely $node $ClusterNetwork
}

if (Test-ContainerExists $ClientContainer) {
    docker rm -f $ClientContainer | Out-Null
}
if (Test-Path $CoordDirectory) {
    Remove-Item -Path $CoordDirectory -Recurse -Force
}
New-Item -Path $CoordDirectory -ItemType Directory -Force | Out-Null

docker run -d `
    --name $ClientContainer `
    --network $ClusterNetwork `
    -v "${PWD}:/app" `
    -w /app `
    python:3.11-slim `
    bash -c "pip install --quiet pymongo && python -u test_mw_failure.py" `
    | Out-Null

$totalStepsFile = Join-Path $CoordDirectory "total_steps.txt"
Wait-ForFile $totalStepsFile $MaximumWaitSeconds
$totalSteps = [int](Get-Content $totalStepsFile).Trim()
Write-Host "Running $totalSteps primary-failure steps."

try {
    for ($step = 1; $step -le $totalSteps; $step++) {
        $stepText = "{0:D3}" -f $step
        $primaryFile = Join-Path $CoordDirectory "step_${stepText}_primary.txt"
        $stopReady = Join-Path $CoordDirectory "step_${stepText}_stop_ready.signal"
        $primaryStopped = Join-Path $CoordDirectory "step_${stepText}_primary_stopped.signal"
        $restartReady = Join-Path $CoordDirectory "step_${stepText}_restart_ready.signal"
        $primaryRestarted = Join-Path $CoordDirectory "step_${stepText}_primary_restarted.signal"
        $doneFile = Join-Path $CoordDirectory "step_${stepText}_done.signal"

        Wait-ForFile $stopReady $MaximumWaitSeconds
        Wait-ForFile $primaryFile $MaximumWaitSeconds
        $oldPrimary = (Get-Content $primaryFile).Trim()
        if ($oldPrimary -notin $Nodes) { throw "Invalid primary name: $oldPrimary" }

        Write-Host "Step $step/${totalSteps}: stopping $oldPrimary..."
        docker stop $oldPrimary | Out-Null
        New-Signal $primaryStopped

        Wait-ForFile $restartReady $MaximumWaitSeconds
        Write-Host "Restarting $oldPrimary..."
        docker start $oldPrimary | Out-Null
        New-Signal $primaryRestarted
        Wait-ForFile $doneFile $MaximumWaitSeconds
    }

    $completeFile = Join-Path $CoordDirectory "experiment_complete.signal"
    Wait-ForFile $completeFile $MaximumWaitSeconds
    $exitCode = (docker wait $ClientContainer).Trim()
    docker logs $ClientContainer
    if ($exitCode -ne "0") { throw "Python test exited with code $exitCode" }
}
finally {
    foreach ($node in $Nodes) {
        if (Test-ContainerExists $node) {
            if (-not (Test-ContainerRunning $node)) { docker start $node | Out-Null }
            Connect-Safely $node $ClusterNetwork
        }
    }
    if (Test-ContainerExists $ClientContainer) {
        if (Test-ContainerRunning $ClientContainer) { docker stop $ClientContainer | Out-Null }
        docker rm $ClientContainer | Out-Null
    }
}

Write-Host "Primary-failure experiment completed."
