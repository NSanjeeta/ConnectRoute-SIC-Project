param(
    [int]$TuningTrials = 50,
    [int]$BootstrapReplicates = 2000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project virtual-environment interpreter not found: $python"
}
if ($TuningTrials -lt 1) {
    throw "TuningTrials must be at least one."
}
if ($BootstrapReplicates -lt 1) {
    throw "BootstrapReplicates must be at least one."
}

Push-Location $projectRoot
try {
    function Invoke-PipelineStage {
        param([string[]]$Arguments)

        Write-Host "`n>>> $python $($Arguments -join ' ')"
        & $python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Pipeline stage failed with exit code $($LASTEXITCODE): $($Arguments -join ' ')"
        }
    }

    Invoke-PipelineStage @("-m", "connectroute.ingest")
    Invoke-PipelineStage @("-m", "connectroute.tower_ingest")
    Invoke-PipelineStage @("-m", "connectroute.geometry")
    Invoke-PipelineStage @("-m", "connectroute.towers")
    Invoke-PipelineStage @("-m", "connectroute.terrain")
    Invoke-PipelineStage @("-m", "connectroute.propagation")
    Invoke-PipelineStage @("-m", "connectroute.models", "--trials", "$TuningTrials")
    Invoke-PipelineStage @(
        "-m",
        "connectroute.evaluate",
        "--bootstrap-replicates",
        "$BootstrapReplicates"
    )
}
finally {
    Pop-Location
}
