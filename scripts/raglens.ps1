# Windows PowerShell task runner (equivalent of the Makefile).
# Usage: .\scripts\raglens.ps1 install | install-vectors | lint | test | eval | serve | ui
param([Parameter(Mandatory = $true)][string]$Task)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

switch ($Task) {
    "install" {
        if (-not (Test-Path ".venv")) { py -3.11 -m venv .venv }
        .\.venv\Scripts\python.exe -m pip install --upgrade pip
        .\.venv\Scripts\pip.exe install -e ".[dev]"
    }
    "install-vectors" { .\.venv\Scripts\pip.exe install -e ".[dev,vectors]" }
    "lint" {
        .\.venv\Scripts\ruff.exe check .
        .\.venv\Scripts\ruff.exe format --check .
        .\.venv\Scripts\mypy.exe src
    }
    "test" { .\.venv\Scripts\pytest.exe --cov=raglens --cov-report=term-missing tests/ }
    "eval" { .\.venv\Scripts\raglens.exe eval }
    "serve" { .\.venv\Scripts\raglens.exe serve }
    "ui" { .\.venv\Scripts\streamlit.exe run src/raglens/ui/app.py }
    default { throw "Unknown task '$Task'" }
}
