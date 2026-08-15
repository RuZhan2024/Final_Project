$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    $python = "python"
}

Push-Location $root
try {
    & $python -m unittest discover -s packages/ml/tests
    & $python -m unittest discover -s applications/api/tests

    Push-Location applications/web
    try {
        npm run typecheck
        npm run build
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
