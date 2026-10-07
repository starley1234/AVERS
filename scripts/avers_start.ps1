# AVERS Web UI - запуск в фоне (Windows).
# Использование: start.cmd [-Port 8030] [-NoBrowser]
param(
    [int]$Port = 8030,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
$PidFile = Join-Path $Root ".avers_web.pid"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

# Уже запущен?
if (Test-Path $PidFile) {
    $old = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($old -and (Get-Process -Id $old -ErrorAction SilentlyContinue)) {
        Write-Host "AVERS уже запущен (PID $old): http://127.0.0.1:$Port" -ForegroundColor Yellow
        if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port" }
        exit 0
    }
    Remove-Item $PidFile -Force
}
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Порт $Port занят процессом PID $($busy[0].OwningProcess). Остановите его (stop.cmd) или укажите -Port." -ForegroundColor Red
    exit 1
}

# Окружение: создаём venv и ставим зависимости при первом запуске.
if (-not (Test-Path $Python)) {
    Write-Host "Создаю виртуальное окружение venv (Python 3.12)..."
    & py -3.12 -m venv venv
    if ($LASTEXITCODE -ne 0) { & python -m venv venv }
    & $Python -m pip install --upgrade pip
    & $Python -m pip install -r requirements.txt
}
& $Python -c "import cv2, fastapi, uvicorn, skimage" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Доустанавливаю зависимости из requirements.txt..."
    & $Python -m pip install -r requirements.txt
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $LogDir "avers_web_$stamp.log"
$err = Join-Path $LogDir "avers_web_$stamp.err.log"
$env:PYTHONIOENCODING = "utf-8"
$proc = Start-Process -FilePath $Python `
    -ArgumentList @("-m", "avers", "web", "--host", "127.0.0.1", "--port", "$Port") `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
Set-Content -Path $PidFile -Value $proc.Id

# Ждём, пока сервер ответит.
$url = "http://127.0.0.1:$Port"
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) { break }
    try {
        $r = Invoke-WebRequest -Uri "$url/api/gost/library" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) {
    Write-Host "Сервер не запустился. Лог: $err" -ForegroundColor Red
    if (Test-Path $err) { Get-Content $err -Tail 20 }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "AVERS запущен: $url  (PID $($proc.Id))" -ForegroundColor Green
Write-Host "Логи: $LogDir   Остановить: stop.cmd"
Write-Host "Демо: кнопка '▶ Демо-схема БКС (ГОСТ)' -> 'Запустить обработку'"
if (-not $NoBrowser) { Start-Process $url }
