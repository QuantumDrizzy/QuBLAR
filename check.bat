@echo off
setlocal
cd /d "%~dp0"

rem Every check, in the order a failure is cheapest to understand: the tracer, then the
rem physics on top of it, then the two tracers against each other. Exit non-zero if any
rem of them fails, so this is usable without reading the output.

build\trace.exe          || exit /b 1
build\check_lidar.exe    || exit /b 1
build\check_detector.exe || exit /b 1
build\bench_trace.exe build\optix_programs.optixir || exit /b 1

echo.
echo all green
endlocal
