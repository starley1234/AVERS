@echo off
rem AVERS Web UI - stop. Usage: stop.cmd [-Port 8030]
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\avers_stop.ps1" %*
