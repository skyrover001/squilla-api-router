# Squilla API Router - Windows background starter.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File start_router.ps1          # 8021
#   powershell -ExecutionPolicy Bypass -File start_router.ps1 8022    # custom
#
# The service runs detached (Start-Process -WindowStyle Hidden), so closing
# the terminal does NOT stop it.  Logs:
#   stdout -> uvicorn_<port>.log, stderr -> uvicorn_<port>.err.log

param(
    [int]$Port = 8021
)

$ErrorActionPreference = "Stop"

$RunDir  = "C:\Users\skyro\AppData\Local\Temp\squilla_api_router"
$SrcDir  = $PSScriptRoot
$RootDir = Split-Path -Parent $SrcDir
$Python  = "D:\Program Files\python\python.exe"
$Listen  = "127.0.0.1"

# 1) Sync runtime files into the ASCII run dir.  LightGBM cannot open its
#    model bundle from a non-ASCII (Chinese) path, so app.py + model_bundle
#    are copied here; the repo stays on sys.path via _run_server.py.
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
foreach ($f in @("app.py", ".env", "_formats_rosetta.py",
                 "_cd_translate.py", "_cd_sse.py", "_cd_log.py")) {
    Copy-Item "$SrcDir\$f" "$RunDir\$f" -Force
}
Copy-Item "$SrcDir\_run_server.py" "$RunDir\_run_server.py" -Force
if (-not (Test-Path "$RunDir\model_bundle\router.runtime.yaml")) {
    Copy-Item "$SrcDir\model_bundle\*" "$RunDir\model_bundle\" -Recurse -Force
}

# 2) Stop any previous instance listening on this port.
$old = netstat -ano | Select-String ":$Port\s" | Select-Object -First 1
if ($old) {
    $oldPid = ($old.ToString().Trim() -split '\s+') | Select-Object -Last 1
    Write-Host "Stopping previous instance (PID $oldPid)..."
    Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# 3) Start detached.  SQUILLA_REPO is set in THIS PowerShell process and
#    inherited by the child (Start-Process), avoiding cmd encoding issues.
$env:SQUILLA_REPO = $RootDir
$proc = Start-Process -FilePath $Python `
    -ArgumentList "$RunDir\_run_server.py", "--host", "$Listen", "--port", "$Port" `
    -WorkingDirectory $RunDir `
    -RedirectStandardOutput "$RunDir\uvicorn_$Port.log" `
    -RedirectStandardError  "$RunDir\uvicorn_$Port.err.log" `
    -WindowStyle Hidden -PassThru

# 4) Wait for health.
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod -Uri "http://$Listen`:$Port/health" -TimeoutSec 3
        if ($h.status -eq "ok") { $ok = $true; break }
    } catch {}
}

if ($ok) {
    Write-Host "Squilla Router is running:" -ForegroundColor Green
    Write-Host "  PID:   $($proc.Id)"
    Write-Host "  URL:   http://$Listen`:$Port/v1"
    Write-Host "  ML:    ml_ready=$($h.ml_ready)"
    Write-Host "  Logs:  $RunDir\uvicorn_$Port.log / .err.log"
} else {
    Write-Host "FAILED to start. Check $RunDir\uvicorn_$Port.err.log" -ForegroundColor Red
    exit 1
}
