param(
    [string]$Python = "",
    [string]$VenvDir = ".venv-win",
    [switch]$SkipStart
)

$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    $scriptPath = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $scriptPath "..\..")).Path
}

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-Python {
    param(
        [string]$PythonExe,
        [string[]]$Arguments
    )
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $PythonExe $($Arguments -join ' ')"
    }
}

function Test-PythonCompatible {
    param([string]$PythonExe)
    $code = "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    & $PythonExe -c $code *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-Python {
    param([string]$RequestedPython)

    if ($RequestedPython.Trim()) {
        if (Test-PythonCompatible $RequestedPython) {
            return $RequestedPython
        }
        throw "$RequestedPython must be Python 3.10 or newer."
    }

    $candidates = @(
        @("py", @("-3.10")),
        @("py", @("-3")),
        @("python", @()),
        @("python3", @())
    )

    foreach ($candidate in $candidates) {
        $cmd = [string]$candidate[0]
        $prefixArgs = [string[]]$candidate[1]
        if (-not (Test-CommandExists $cmd)) {
            continue
        }
        $code = "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
        & $cmd @prefixArgs -c $code *> $null
        if ($LASTEXITCODE -eq 0) {
            if ($prefixArgs.Count -gt 0) {
                return "$cmd $($prefixArgs -join ' ')"
            }
            return $cmd
        }
    }

    throw "Could not find Python 3.10+. Install Python 3.10/3.11, or rerun with: -Python C:\Path\To\python.exe"
}

function Invoke-PythonCommand {
    param(
        [string]$PythonCommand,
        [string[]]$Arguments
    )
    $parts = $PythonCommand -split " "
    $exe = $parts[0]
    $prefix = @()
    if ($parts.Count -gt 1) {
        $prefix = $parts[1..($parts.Count - 1)]
    }
    & $exe @prefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $PythonCommand $($Arguments -join ' ')"
    }
}

$root = Get-ProjectRoot
Set-Location $root

$localNode = Join-Path $root ".tools\node-v22-win-x64"
if (Test-Path (Join-Path $localNode "npm.cmd")) {
    $env:PATH = "$localNode;$env:PATH"
}
$npmCmd = Join-Path $localNode "npm.cmd"
if (-not (Test-Path $npmCmd)) {
    $npmCmd = "npm.cmd"
}

$pythonCmd = Resolve-Python $Python

if (-not (Test-CommandExists "npm.cmd")) {
    throw "Missing npm. Install Node.js 22.x LTS, then open a new PowerShell window."
}

$nodeMajor = ""
if (Test-CommandExists "node") {
    $nodeMajor = (& node -p "process.versions.node.split('.')[0]" 2>$null)
}
if ($nodeMajor -and $nodeMajor -ne "22") {
    Write-Warning "Detected Node.js $(node --version). Node.js 22.x is recommended for frontend parity."
}

$venvPython = Join-Path $root "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[bootstrap] creating $VenvDir"
    Invoke-PythonCommand $pythonCmd @("-m", "venv", $VenvDir)
}

Write-Host "[bootstrap] upgrading packaging tools"
Invoke-Python $venvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

$depsOk = $false
try {
    $depProbe = Start-Process -FilePath $venvPython -ArgumentList @("-c", "import fastapi, uvicorn, yaml, numpy, torch") -Wait -PassThru -NoNewWindow -RedirectStandardOutput "$env:TEMP\fall_detection_dep_probe.out" -RedirectStandardError "$env:TEMP\fall_detection_dep_probe.err"
    if ($depProbe.ExitCode -eq 0) {
        $depsOk = $true
    }
}
catch {
    $depsOk = $false
}

if (-not $depsOk) {
    Write-Host "[bootstrap] installing backend/runtime dependencies"
    try {
        Invoke-Python $venvPython @("-m", "pip", "install", "-r", "requirements.txt")
    }
    catch {
        Write-Warning "Full requirements install failed on Windows. Falling back to requirements_server.txt for app startup."
        Invoke-Python $venvPython @("-m", "pip", "install", "-r", "requirements_server.txt")
    }
    Invoke-Python $venvPython @("-m", "pip", "install", "-e", ".", "--no-build-isolation")
}

$frontendDir = Join-Path $root "applications\frontend"
$nodeModules = Join-Path $frontendDir "node_modules"
if (-not (Test-Path $nodeModules)) {
    Write-Host "[bootstrap] installing frontend dependencies"
    Push-Location $frontendDir
    try {
        & $npmCmd install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed"
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "[bootstrap] syncing frontend MediaPipe assets"
Push-Location $frontendDir
try {
    & $npmCmd run sync-mediapipe-assets
    if ($LASTEXITCODE -ne 0) {
        throw "npm run sync-mediapipe-assets failed"
    }
}
finally {
    Pop-Location
}

if ($SkipStart) {
    Write-Host "[bootstrap] done. Start later with: powershell -ExecutionPolicy Bypass -File ops\scripts\start_fullstack.ps1"
    exit 0
}

& (Join-Path $root "ops\scripts\start_fullstack.ps1") -VenvDir $VenvDir
