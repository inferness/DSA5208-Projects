param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("NP1", "NP2", "NP3", "NP4", "NP5")]
    [string]$Case
)

$ErrorActionPreference = "Stop"

$ClientContainer = "mw-partition-client"
$ClusterNetwork = "mongoCluster"
$CoordDirectory = Join-Path $PWD "mw_partition_coord"
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


function Test-NetworkExists {
    param([string]$Name)
    $found = docker network ls --filter "name=^$Name$" --format "{{.Name}}"
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
        if ($LASTEXITCODE -ne 0) {
            throw "Could not connect $Container to $Network"
        }
    }
}


function Disconnect-Safely {
    param([string]$Container, [string]$Network)
    if (Test-ConnectedToNetwork $Container $Network) {
        docker network disconnect --force $Network $Container
        if ($LASTEXITCODE -ne 0) {
            throw "Could not disconnect $Container from $Network"
        }
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
            Write-Host "Python container stopped unexpectedly."
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


function Remove-NetworkSafely {
    param([string]$Name)
    if (Test-NetworkExists $Name) {
        docker network rm $Name | Out-Null
    }
}


function Heal-StepNetworks {
    param([string]$MinorityNetwork, [string]$MajorityNetwork)

    foreach ($node in $Nodes) {
        if (Test-ContainerExists $node) {
            if (-not (Test-ContainerRunning $node)) {
                docker start $node | Out-Null
            }
            Connect-Safely $node $ClusterNetwork
        }
    }

    foreach ($node in $Nodes) {
        if (Test-NetworkExists $MinorityNetwork) {
            Disconnect-Safely $node $MinorityNetwork
        }
        if (Test-NetworkExists $MajorityNetwork) {
            Disconnect-Safely $node $MajorityNetwork
        }
    }

    if (Test-ContainerExists $ClientContainer) {
        if (Test-NetworkExists $MinorityNetwork) {
            Disconnect-Safely $ClientContainer $MinorityNetwork
        }
        if (Test-NetworkExists $MajorityNetwork) {
            Disconnect-Safely $ClientContainer $MajorityNetwork
        }
    }

    Remove-NetworkSafely $MinorityNetwork
    Remove-NetworkSafely $MajorityNetwork
}


foreach ($node in $Nodes) {
    if (-not (Test-ContainerExists $node)) {
        throw "$node does not exist. Run 'docker compose up -d' first."
    }
    if (-not (Test-ContainerRunning $node)) {
        docker start $node | Out-Null
    }
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
    -e "SELECTED_CASE=$Case" `
    -w /app `
    python:3.11-slim `
    bash -c "pip install --quiet pymongo && python -u test_mw_partition.py" `
    | Out-Null

$totalStepsFile = Join-Path $CoordDirectory "total_steps.txt"
Wait-ForFile $totalStepsFile $MaximumWaitSeconds
$totalSteps = [int](Get-Content $totalStepsFile).Trim()

Write-Host "Running MW ${Case}: $totalSteps network-partition steps."

try {
    for ($step = 1; $step -le $totalSteps; $step++) {
        $stepText = "{0:D3}" -f $step
        $minorityNetwork = "mwMinority$stepText"
        $majorityNetwork = "mwMajority$stepText"

        $primaryFile = Join-Path $CoordDirectory "step_${stepText}_primary.txt"
        $partitionReady = Join-Path $CoordDirectory "step_${stepText}_partition_ready.signal"
        $partitionCreated = Join-Path $CoordDirectory "step_${stepText}_partition_created.signal"
        $healReady = Join-Path $CoordDirectory "step_${stepText}_heal_ready.signal"
        $partitionHealed = Join-Path $CoordDirectory "step_${stepText}_partition_healed.signal"
        $doneFile = Join-Path $CoordDirectory "step_${stepText}_done.signal"
        $majorityFile = Join-Path $CoordDirectory "step_${stepText}_majority.txt"

        Wait-ForFile $partitionReady $MaximumWaitSeconds
        Wait-ForFile $primaryFile $MaximumWaitSeconds
        $oldPrimary = (Get-Content $primaryFile).Trim()

        if ($oldPrimary -notin $Nodes) {
            throw "Invalid primary name: $oldPrimary"
        }

        $secondaries = @($Nodes | Where-Object { $_ -ne $oldPrimary })
        $minoritySecondary = $secondaries[0]
        $majorityNodes = @($secondaries[1], $secondaries[2], $secondaries[3])

        Set-Content -Path $majorityFile -Value $majorityNodes -Encoding ASCII

        Write-Host ""
        Write-Host "Step $step/$totalSteps"
        Write-Host "Minority: $oldPrimary, $minoritySecondary"
        Write-Host "Majority: $($majorityNodes -join ', ')"

        docker network create $minorityNetwork | Out-Null
        docker network create $majorityNetwork | Out-Null

        Connect-Safely $ClientContainer $minorityNetwork
        Connect-Safely $ClientContainer $majorityNetwork
        Connect-Safely $oldPrimary $minorityNetwork
        Connect-Safely $minoritySecondary $minorityNetwork
        foreach ($node in $majorityNodes) {
            Connect-Safely $node $majorityNetwork
        }

        # All alternative paths are ready. Removing mongoCluster now creates
        # the actual database-to-database 2-versus-3 split.
        foreach ($node in $Nodes) {
            Disconnect-Safely $node $ClusterNetwork
        }

        New-Signal $partitionCreated
        Wait-ForFile $healReady $MaximumWaitSeconds

        Write-Host "Healing partition for step $step..."
        Heal-StepNetworks $minorityNetwork $majorityNetwork
        New-Signal $partitionHealed
        Wait-ForFile $doneFile $MaximumWaitSeconds
        Write-Host "Step $step completed."
    }

    $completeFile = Join-Path $CoordDirectory "experiment_complete.signal"
    Wait-ForFile $completeFile $MaximumWaitSeconds
    $exitCode = (docker wait $ClientContainer).Trim()

    Write-Host ""
    Write-Host "============================================================"
    docker logs $ClientContainer
    Write-Host "============================================================"

    if ($exitCode -ne "0") {
        throw "Python test exited with code $exitCode"
    }
}
catch {
    Write-Host ""
    Write-Host "Experiment failed. Last Python output:"
    if (Test-ContainerExists $ClientContainer) {
        docker logs --tail 100 $ClientContainer
    }
    throw
}
finally {
    # Recover the standard topology even if the experiment fails midway.
    foreach ($node in $Nodes) {
        if (Test-ContainerExists $node) {
            if (-not (Test-ContainerRunning $node)) {
                docker start $node | Out-Null
            }
            Connect-Safely $node $ClusterNetwork
        }
    }

    $temporaryNetworks = docker network ls --format "{{.Name}}" |
        Where-Object { $_ -like "mwMinority*" -or $_ -like "mwMajority*" }

    foreach ($network in $temporaryNetworks) {
        foreach ($node in $Nodes) {
            Disconnect-Safely $node $network
        }
        if (Test-ContainerExists $ClientContainer) {
            Disconnect-Safely $ClientContainer $network
        }
        Remove-NetworkSafely $network
    }

    if (Test-ContainerExists $ClientContainer) {
        if (Test-ContainerRunning $ClientContainer) {
            docker stop $ClientContainer | Out-Null
        }
        docker rm $ClientContainer | Out-Null
    }
}

Write-Host "Network-partition experiment completed."
