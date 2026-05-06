param(
    [string]$StatePath = ".make\dev-windows.json",
    [int[]]$Ports = @(8000, 3000)
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..\..")).Path
Set-Location $root

function Stop-ProjectProcess {
    param([int]$ProcessId)
    if (-not $ProcessId) {
        return
    }
    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
    }
    catch {
        return
    }
    $cmd = [string]$proc.CommandLine
    $isProject = (
        $cmd -like "*$root*" -or
        $cmd -like "*applications.backend.app:app*" -or
        $cmd -like "*react-scripts*"
    )
    if ($isProject) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "[dev] stopped process $ProcessId"
    }
    else {
        Write-Host "[dev] left non-project process $ProcessId alone"
    }
}

if (Test-Path $StatePath) {
    $state = Get-Content -Raw $StatePath | ConvertFrom-Json
    foreach ($pidValue in @($state.backend_pid, $state.frontend_pid)) {
        if ($pidValue) {
            Stop-ProjectProcess ([int]$pidValue)
        }
    }
    Remove-Item $StatePath -Force -ErrorAction SilentlyContinue
}
else {
    Write-Host "[dev] no Windows dev state file found: $StatePath"
}

foreach ($port in $Ports) {
    $owners = @(
        Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    foreach ($pidValue in $owners) {
        if ($pidValue) {
            Stop-ProjectProcess ([int]$pidValue)
        }
    }
}
Write-Host "[dev] stopped Windows dev processes"
