@echo off
REM Test launcher: run RFMetrics on the Vulkan backend.
REM The exe defaults to DirectX 12 (see main.rs): wgpu's auto-picked
REM Vulkan backend stalls surface presents across window resizes,
REM leaving a persistent black box on the grown area.
set WGPU_BACKEND=vulkan
start "" "%~dp0target\release\rfmetrics.exe"
