@echo off
rem AVERS Web UI - start in background. Usage: start.cmd [-Port 8030] [-NoBrowser]
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\avers_start.ps1" %*
if errorlevel 1 pause
