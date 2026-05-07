param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DevArgs
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..\..")).Path
$venvPython = Join-Path $root ".venv-win\Scripts\python.exe"

if (Test-Path $venvPython) {
    $python = $venvPython
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.10 (Join-Path $root "ops\scripts\dev.py") @DevArgs
    exit $LASTEXITCODE
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = "python"
}
else {
    throw "Missing Python 3.10+. Install Python, then rerun this command."
}

& $python (Join-Path $root "ops\scripts\dev.py") @DevArgs
exit $LASTEXITCODE
