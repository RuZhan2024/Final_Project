param(
    [string]$VenvDir = ".venv-win",
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [string]$FrontendHost = "127.0.0.1",
    [int]$FrontendPort = 3000,
    [string]$Browser = "none",
    [switch]$Detached
)

$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    $scriptPath = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $scriptPath "..\..")).Path
}

function Test-PortFree {
    param(
        [string]$HostName,
        [int]$Port
    )
    $listener = $null
    try {
        $address = [System.Net.IPAddress]::Parse($HostName)
        $listener = [System.Net.Sockets.TcpListener]::new($address, $Port)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Wait-ForHealth {
    param(
        [string]$Url,
        [int]$Attempts = 30
    )
    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2 | Out-Null
            return $true
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
    return $false
}

function Get-ListeningProcessId {
    param([int]$Port)
    try {
        $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -First 1
        if ($conn -and $conn.OwningProcess) {
            return [int]$conn.OwningProcess
        }
    }
    catch {
        return $null
    }
    return $null
}

$root = Get-ProjectRoot
$frontendDir = Join-Path $root "applications\frontend"
$venvPython = Join-Path $root "$VenvDir\Scripts\python.exe"
$nodeModules = Join-Path $frontendDir "node_modules"
$backendLog = Join-Path ([System.IO.Path]::GetTempPath()) "fall_detection_backend.log"
$backendErrLog = Join-Path ([System.IO.Path]::GetTempPath()) "fall_detection_backend.err.log"
$healthUrl = "http://${BackendHost}:${BackendPort}/api/health"
$apiBase = "http://${BackendHost}:${BackendPort}"
$localNode = Join-Path $root ".tools\node-v22-win-x64"

if (Test-Path (Join-Path $localNode "npm.cmd")) {
    $env:PATH = "$localNode;$env:PATH"
}
$npmCmd = Join-Path $localNode "npm.cmd"
if (-not (Test-Path $npmCmd)) {
    $npmCmd = "npm.cmd"
}

if (-not (Test-Path $venvPython)) {
    throw "Missing $venvPython. Run: powershell -ExecutionPolicy Bypass -File ops\scripts\bootstrap_dev.ps1"
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "Missing npm. Install Node.js 22.x LTS, then open a new PowerShell window."
}
if (-not (Test-Path $nodeModules)) {
    throw "Missing frontend node_modules. Run: powershell -ExecutionPolicy Bypass -File ops\scripts\bootstrap_dev.ps1"
}
if (-not (Test-PortFree $BackendHost $BackendPort)) {
    throw "Backend port already in use: ${BackendHost}:${BackendPort}"
}
if (-not (Test-PortFree $FrontendHost $FrontendPort)) {
    throw "Frontend port already in use: ${FrontendHost}:${FrontendPort}"
}

Set-Location $root
$env:PYTHONPATH = "$root\ml\src;$root"

Write-Host "[dev] starting backend on ${BackendHost}:${BackendPort}"
$backendArgs = @(
    "-m", "uvicorn",
    "applications.backend.app:app",
    "--host", $BackendHost,
    "--port", "$BackendPort"
)
$backend = Start-Process -FilePath $venvPython -ArgumentList $backendArgs -WorkingDirectory $root -RedirectStandardOutput $backendLog -RedirectStandardError $backendErrLog -PassThru -WindowStyle Hidden

try {
    if (-not (Wait-ForHealth $healthUrl 30)) {
        throw "Backend failed health check: $healthUrl. Backend logs: $backendLog / $backendErrLog"
    }

    Write-Host "[dev] backend healthy: $healthUrl"
    Write-Host "[dev] backend logs: $backendLog / $backendErrLog"
    Write-Host "[dev] starting frontend on ${FrontendHost}:${FrontendPort}"
    Write-Host "[dev] frontend URL: http://${FrontendHost}:${FrontendPort}"

    Set-Location $frontendDir
    $env:HOST = $FrontendHost
    $env:PORT = "$FrontendPort"
    $env:BROWSER = $Browser
    $env:REACT_APP_API_BASE = $apiBase

    if ($Detached) {
        $frontendOut = Join-Path ([System.IO.Path]::GetTempPath()) "fall_detection_frontend.log"
        $frontendErr = Join-Path ([System.IO.Path]::GetTempPath()) "fall_detection_frontend.err.log"
        $frontend = Start-Process -FilePath $npmCmd -ArgumentList @("start") -WorkingDirectory $frontendDir -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -PassThru -WindowStyle Hidden

        $frontendUrl = "http://${FrontendHost}:${FrontendPort}"
        if (-not (Wait-ForHealth $frontendUrl 60)) {
            throw "Frontend failed readiness check: $frontendUrl. Frontend logs: $frontendOut / $frontendErr"
        }

        $backendPid = Get-ListeningProcessId $BackendPort
        $frontendPid = Get-ListeningProcessId $FrontendPort
        if (-not $backendPid) {
            $backendPid = $backend.Id
        }
        if (-not $frontendPid) {
            $frontendPid = $frontend.Id
        }

        $stateDir = Join-Path $root ".make"
        New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
        $state = [ordered]@{
            backend_pid = $backendPid
            frontend_pid = $frontendPid
            backend_url = $healthUrl
            frontend_url = $frontendUrl
            backend_stdout = $backendLog
            backend_stderr = $backendErrLog
            frontend_stdout = $frontendOut
            frontend_stderr = $frontendErr
        }
        $state | ConvertTo-Json | Set-Content -Path (Join-Path $stateDir "dev-windows.json") -Encoding UTF8
        Write-Host "[dev] detached processes started. State: .make\dev-windows.json"
        return
    }
    else {
        & $npmCmd start
        if ($LASTEXITCODE -ne 0) {
            throw "npm start failed"
        }
    }
}
finally {
    if (-not $Detached -and $backend -and -not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}
