<#
.SYNOPSIS
  One-step setup on Windows: creates .venv (Python 3.12, else 3.11), installs requirements,
  and copies .env.example to .env if .env does not exist yet.

.USAGE
  From the project folder, in PowerShell:
      powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
  Optional: force a version
      powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -PythonVersion 3.11

  "-ExecutionPolicy Bypass" applies to this single command only; no system-wide change is made.
#>
param([string]$PythonVersion = "")

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "The 'py' launcher was not found. Install Python 3.11 or 3.12 from python.org (tick 'py launcher')." -ForegroundColor Red
    exit 1
}

$candidates = @("3.12", "3.11")
if ($PythonVersion) { $candidates = @($PythonVersion) }

$chosen = $null
foreach ($v in $candidates) {
    & py "-$v" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $chosen = $v; break }
}
if (-not $chosen) {
    Write-Host "Python $($candidates -join ' / ') not found. Run 'py -0' to list installed versions." -ForegroundColor Red
    exit 1
}

Write-Host "Using Python $chosen"
& py "-$chosen" -m venv .venv
if ($LASTEXITCODE -ne 0) { exit 1 }

$venvPython = Join-Path ".venv" "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env - open it and paste your Groq API key after GROQ_API_KEY=" -ForegroundColor Yellow
} else {
    Write-Host ".env already exists - left unchanged."
}

Write-Host ""
Write-Host "Setup complete. Next steps (no need to activate the venv if you use these exact commands):"
Write-Host "  .\.venv\Scripts\python.exe run_eval.py                 # deterministic evaluation, no API key needed"
Write-Host "  .\.venv\Scripts\python.exe -m pytest                   # tests, no API key needed"
Write-Host "  .\.venv\Scripts\python.exe run_agent.py                # live Groq chat (needs the key in .env)"
Write-Host "  .\.venv\Scripts\python.exe run_agent.py --simulated    # offline chat, no key"
