$ErrorActionPreference = "Stop"

$ClientContainer = "mw-failure-client"
$CoordDirectory = Join-Path $PWD "mw_coord"
$MaximumWaitSeconds = 240


function Test-ContainerRunning {
    param(
        [string]$ContainerName
    )

    $runningContainers = docker ps `
        --filter "name=^/$ContainerName$" `
        --format "{{.Names}}"

    return (
        $runningContainers -contains $ContainerName
    )
}


function Test-ContainerExists {
    param(
        [string]$ContainerName
    )

    $containers = docker ps -a `
        --filter "name=^/$ContainerName$" `
        --format "{{.Names}}"

    return (
        $containers -contains $ContainerName
    )
}


function Wait-ForSignalFile {
    param(
        [string]$Path,
        [int]$TimeoutSeconds = 240
    )

    $startTime = Get-Date

    while (-not (Test-Path $Path)) {
        $elapsed = (
            (Get-Date) - $startTime
        ).TotalSeconds

        if ($elapsed -ge $TimeoutSeconds) {
            throw "Timed out waiting for file: $Path"
        }

        if (
            (Test-ContainerExists $ClientContainer) -and
            -not (Test-ContainerRunning $ClientContainer)
        ) {
            Write-Host ""
            Write-Host "Python container stopped unexpectedly."
            Write-Host "Python output:"
            docker logs $ClientContainer

            throw "The Python test exited before creating: $Path"
        }

        Start-Sleep -Milliseconds 250
    }
}


function New-SignalFile {
    param(
        [string]$Path
    )

    Set-Content `
        -Path $Path `
        -Value "ready" `
        -Encoding ASCII
}


function Confirm-MongoContainers {
    $mongoContainers = @(
        "mongo1",
        "mongo2",
        "mongo3",
        "mongo4",
        "mongo5"
    )

    foreach ($container in $mongoContainers) {
        if (-not (Test-ContainerExists $container)) {
            throw (
                "Container $container does not exist. " +
                "Run 'docker compose up -d' first."
            )
        }

        if (-not (Test-ContainerRunning $container)) {
            Write-Host "Starting $container..."
            docker start $container | Out-Null
        }
    }

    Write-Host "All five MongoDB containers are running."
}


# ------------------------------------------------------------
# Preparation
# ------------------------------------------------------------

Write-Host ""
Write-Host "Preparing monotonic-writes failure experiment..."
Write-Host ""

Confirm-MongoContainers

# Remove only the old temporary test-client container.
if (Test-ContainerExists $ClientContainer) {
    Write-Host "Removing previous temporary test container..."
    docker rm -f $ClientContainer | Out-Null
}

# Remove only the experiment's coordination directory.
if (Test-Path $CoordDirectory) {
    Remove-Item `
        -Path $CoordDirectory `
        -Recurse `
        -Force
}

New-Item `
    -Path $CoordDirectory `
    -ItemType Directory `
    -Force | Out-Null


# ------------------------------------------------------------
# Start Python test in a background Docker container
# ------------------------------------------------------------

Write-Host "Starting Python test container..."

docker run -d `
    --name $ClientContainer `
    --network mongoCluster `
    -v "${PWD}:/app" `
    -w /app `
    python:3.11-slim `
    bash -c "pip install --quiet pymongo && python -u test_mw_failure.py" `
    | Out-Null

$totalStepsFile = Join-Path `
    $CoordDirectory `
    "total_steps.txt"

Write-Host "Waiting for Python test to initialize..."

Wait-ForSignalFile `
    -Path $totalStepsFile `
    -TimeoutSeconds $MaximumWaitSeconds

$totalSteps = [int](
    Get-Content $totalStepsFile
).Trim()

Write-Host "Python test initialized."
Write-Host "Total primary-failure steps: $totalSteps"
Write-Host ""


# ------------------------------------------------------------
# Coordinate each primary failure
# ------------------------------------------------------------

for ($step = 1; $step -le $totalSteps; $step++) {
    $stepText = "{0:D3}" -f $step

    $primaryFile = Join-Path `
        $CoordDirectory `
        "step_${stepText}_primary.txt"

    $stopReadyFile = Join-Path `
        $CoordDirectory `
        "step_${stepText}_stop_ready.signal"

    $primaryStoppedFile = Join-Path `
        $CoordDirectory `
        "step_${stepText}_primary_stopped.signal"

    $restartReadyFile = Join-Path `
        $CoordDirectory `
        "step_${stepText}_restart_ready.signal"

    $primaryRestartedFile = Join-Path `
        $CoordDirectory `
        "step_${stepText}_primary_restarted.signal"

    $doneFile = Join-Path `
        $CoordDirectory `
        "step_${stepText}_done.signal"

    Write-Host "--------------------------------------------"
    Write-Host "Failure step $step of $totalSteps"
    Write-Host "--------------------------------------------"

    # Wait until W1 has been acknowledged.
    Wait-ForSignalFile `
        -Path $stopReadyFile `
        -TimeoutSeconds $MaximumWaitSeconds

    Wait-ForSignalFile `
        -Path $primaryFile `
        -TimeoutSeconds $MaximumWaitSeconds

    $primaryName = (
        Get-Content $primaryFile
    ).Trim()

    if (
        $primaryName -notin @(
            "mongo1",
            "mongo2",
            "mongo3",
            "mongo4",
            "mongo5"
        )
    ) {
        throw "Invalid primary container name: $primaryName"
    }

    Write-Host "W1 has been acknowledged."
    Write-Host "Current primary: $primaryName"
    Write-Host "Stopping $primaryName..."

    # A short timeout makes the failure occur soon after W1.
    docker stop `
        --time 1 `
        $primaryName | Out-Null

    if (Test-ContainerRunning $primaryName) {
        throw "Docker failed to stop $primaryName."
    }

    Write-Host "$primaryName has stopped."

    New-SignalFile `
        -Path $primaryStoppedFile

    Write-Host "Waiting for election and W2..."

    # Python creates this after a new primary acknowledges W2.
    Wait-ForSignalFile `
        -Path $restartReadyFile `
        -TimeoutSeconds $MaximumWaitSeconds

    Write-Host "W2 has been acknowledged."
    Write-Host "Restarting $primaryName..."

    docker start $primaryName | Out-Null

    if (-not (Test-ContainerRunning $primaryName)) {
        throw "Docker failed to restart $primaryName."
    }

    New-SignalFile `
        -Path $primaryRestartedFile

    Write-Host (
        "$primaryName restarted. " +
        "Waiting for verification..."
    )

    Wait-ForSignalFile `
        -Path $doneFile `
        -TimeoutSeconds $MaximumWaitSeconds

    Write-Host "Step $step completed."
    Write-Host ""
}


# ------------------------------------------------------------
# Wait for Python to finish and show its results
# ------------------------------------------------------------

$experimentCompleteFile = Join-Path `
    $CoordDirectory `
    "experiment_complete.signal"

Wait-ForSignalFile `
    -Path $experimentCompleteFile `
    -TimeoutSeconds $MaximumWaitSeconds

$pythonExitCode = (
    docker wait $ClientContainer
).Trim()

Write-Host ""
Write-Host "============================================"
Write-Host "PYTHON TEST OUTPUT"
Write-Host "============================================"
docker logs $ClientContainer
Write-Host "============================================"

if ($pythonExitCode -ne "0") {
    throw (
        "The Python test exited with code " +
        "$pythonExitCode."
    )
}

docker rm $ClientContainer | Out-Null

Write-Host ""
Write-Host "Experiment finished successfully."
Write-Host "All failed MongoDB containers were restarted."