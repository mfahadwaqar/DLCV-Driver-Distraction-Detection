param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& $Python -c "import sys; assert sys.version_info[:2] == (3,11), 'Python 3.11 required'"
if ($LASTEXITCODE -ne 0) { throw 'Pass -Python with the path to Python 3.11.' }
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& ./.venv/Scripts/python.exe -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& ./.venv/Scripts/python.exe --version | Set-Content docs/environment-phase0.txt
& ./.venv/Scripts/python.exe -m pip freeze | Add-Content docs/environment-phase0.txt
if ($LASTEXITCODE -ne 0) { throw 'Version logging failed' }
