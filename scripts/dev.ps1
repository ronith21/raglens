<#
.SYNOPSIS
  RAGLens task runner for Windows PowerShell (the equivalent of the Makefile).

.EXAMPLE
  .\scripts\dev.ps1 install
  .\scripts\dev.ps1 check
  .\scripts\dev.ps1 eval-real      # needs Ollama running and the [models] extra installed
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "install-models", "lint", "test", "check", "eval", "eval-banking",
                 "gate", "eval-real", "ingest", "serve", "ui", "demo", "screenshots", "help")]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Step([string]$Command) {
    Write-Host "> $Command" -ForegroundColor Cyan
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Command" }
}

switch ($Task) {
    "install"        { Invoke-Step 'python -m pip install -e ".[dev,screenshots]"' }
    "install-models" { Invoke-Step 'python -m pip install -e ".[dev,screenshots,models]"' }
    "lint" {
        Invoke-Step "ruff check ."
        Invoke-Step "ruff format --check ."
        Invoke-Step "mypy src"
    }
    "test"  { Invoke-Step "pytest --cov=raglens --cov-report=term-missing tests/" }
    "check" {
        & $PSCommandPath lint
        & $PSCommandPath test
    }
    "eval"  { Invoke-Step "python -m raglens.eval.benchmark" }
    "eval-banking" {
        Invoke-Step ("python -m raglens.eval.benchmark --corpus data/corpora/banking " +
                     "--dataset data/eval/banking_test.jsonl --out docs/proof/banking")
    }
    "gate"  { Invoke-Step "python -m raglens.eval.train_gate" }
    "eval-real" {
        # Real embeddings + a local LLM through Ollama, on the held-out banking questions.
        Invoke-Step "python -m raglens.eval.train_gate"
        Invoke-Step ("python -m raglens.eval.benchmark --corpus data/corpora/banking " +
                     "--dataset data/eval/banking_test.jsonl --out docs/proof/real " +
                     "--chunk-sizes 512 --configs bm25,dense,hybrid --llm ollama")
    }
    "ingest" { Invoke-Step "raglens ingest" }
    "serve"  { Invoke-Step "uvicorn raglens.api.main:app --port 8000" }
    "ui"     { Invoke-Step "streamlit run src/raglens/ui/app.py" }
    "demo" {
        # Single process: the UI runs the API in-process on the banking corpus.
        $env:API_URL = "embedded"
        $env:CORPUS_DIR = "data/corpora/banking"
        $env:INDEX_DIR = "data/index_banking"
        Invoke-Step "streamlit run src/raglens/ui/app.py"
    }
    "screenshots" { Invoke-Step "python scripts/capture_screenshots.py" }
    default {
        Write-Host "Usage: .\scripts\dev.ps1 <task>"
        Write-Host "Tasks: install, install-models, lint, test, check, eval, eval-banking, gate,"
        Write-Host "       eval-real, ingest, serve, ui, demo, screenshots"
    }
}
