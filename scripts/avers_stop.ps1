# AVERS Web UI - остановка (Windows).
param([int]$Port = 8030)
$Root = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root ".avers_web.pid"
$stopped = $false

if (Test-Path $PidFile) {
    $procId = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        # /T - вместе с дочерними процессами (uvicorn reload/workers)
        & taskkill /PID $procId /T /F | Out-Null
        Write-Host "AVERS остановлен (PID $procId)" -ForegroundColor Green
        $stopped = $true
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

# Запасной вариант: процесс AVERS, слушающий порт (например, запущенный через run_web.cmd).
$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId = $($c.OwningProcess)" -ErrorAction SilentlyContinue
    if ($p -and $p.CommandLine -match "avers") {
        & taskkill /PID $c.OwningProcess /T /F | Out-Null
        Write-Host "Остановлен процесс AVERS на порту $Port (PID $($c.OwningProcess))" -ForegroundColor Green
        $stopped = $true
    } elseif ($p) {
        Write-Host "Порт $Port занят не AVERS: $($p.Name) (PID $($c.OwningProcess)) - не трогаю" -ForegroundColor Yellow
    }
}
if (-not $stopped) { Write-Host "AVERS не был запущен." }
