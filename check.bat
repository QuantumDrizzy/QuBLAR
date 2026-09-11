@echo off
setlocal
cd /d "%~dp0"

rem Every check, in the order a failure is cheapest to understand: the tracer, the sensor
rem built on it, the detector built on that, multi-bounce transport, and finally the two
rem tracers against each other. Exit non-zero if any of them fails, so this is usable
rem without reading the output.

rem Architecture first, because everything below reports numbers that are only meaningful
rem if they came from the hardware they claim to. -arch=sm_120 is a request; this is the
rem confirmation that the request was honoured.
for %%B in (trace check_lidar check_detector check_transient bench_trace) do (
    cuobjdump -lelf build\%%B.exe | findstr /C:"sm_120" >nul || (
        echo ARCH CHECK FAILED: build\%%B.exe contains no sm_120 code
        exit /b 1
    )
)
echo arch sm_120 confirmed in all binaries

build\trace.exe          || exit /b 1
build\check_lidar.exe    || exit /b 1
build\check_detector.exe || exit /b 1
build\check_transient.exe || exit /b 1
build\bench_trace.exe build\optix_programs.optixir || exit /b 1

echo.
echo all green
endlocal
