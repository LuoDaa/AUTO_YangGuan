$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[AUTO_YangGuan] preparing Python environment..." -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  py -3 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Write-Host "[AUTO_YangGuan] preparing frontend..." -ForegroundColor Cyan
Push-Location frontend
if (-not (Test-Path "node_modules")) {
  npm install
}
Pop-Location

Write-Host "[AUTO_YangGuan] starting backend on http://127.0.0.1:8000" -ForegroundColor Green
$backend = Start-Process powershell -PassThru -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$PSScriptRoot'; & '.\.venv\Scripts\python.exe' -m uvicorn autocrown.web:app --host 127.0.0.1 --port 8000 --reload"
)

Start-Sleep -Seconds 2
Write-Host "[AUTO_YangGuan] starting frontend on http://127.0.0.1:5173" -ForegroundColor Green
$frontend = Start-Process powershell -PassThru -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$PSScriptRoot\frontend'; npm run dev"
)

Write-Host ""
Write-Host "Open: http://127.0.0.1:5173" -ForegroundColor Yellow
Write-Host "Backend PID: $($backend.Id)  Frontend PID: $($frontend.Id)"
Write-Host "Close the two spawned PowerShell windows to stop the dev servers."
