@echo off
setlocal
cd /d "%~dp0"

rem ===========================================================================
rem QuBLAR -- Windows build (ADR-002)
rem ===========================================================================
rem Everything is built here, by one compiler, on purpose. OptiX cannot run under
rem WSL on this machine -- libnvoptix.so.1 there is a 14 KB stub -- and the baseline
rem and the accelerated path must not be compiled by different toolchains, or the
rem speedup carries a confound no amount of repetition removes.
rem
rem This file must keep CRLF line endings. An LF-only .bat fails with "not recognized
rem as an internal or external command", which reads like a missing file and is not.
rem ===========================================================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set "OPTIX=C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "FLAGS=-O3 -arch=sm_120 -std=c++17 -lineinfo"

if not exist build mkdir build

echo [1/9] trace
nvcc %FLAGS% -o build\trace.exe src\trace.cu || exit /b 1

echo [2/9] check_lidar
nvcc %FLAGS% -o build\check_lidar.exe src\check_lidar.cu || exit /b 1

rem OptiX device programs compile to OptiX-IR, never to PTX: CUDA 13 validates --ptx
rem output with ptxas, which rejects the OptiX intrinsics as unknown symbols.
echo [3/9] optix module
nvcc --optix-ir -rdc=true -arch=sm_120 -std=c++17 -I "%OPTIX%\include" ^
     -o build\optix_programs.optixir src\optix_programs.cu || exit /b 1

rem advapi32 is needed because OptiX's loader reads the registry to find the driver DLL.
echo [4/9] bench_trace
nvcc %FLAGS% -I "%OPTIX%\include" -o build\bench_trace.exe src\bench_trace.cu ^
     "%CUDA%\lib\x64\cuda.lib" advapi32.lib || exit /b 1

echo [5/9] check_detector
nvcc %FLAGS% -o build\check_detector.exe src\check_detector.cu || exit /b 1

echo [6/9] check_transient
nvcc %FLAGS% -o build\check_transient.exe src\check_transient.cu || exit /b 1

echo [7/9] check_external
nvcc %FLAGS% -o build\check_external.exe src\check_external.cu || exit /b 1

echo [8/9] check_muon
nvcc %FLAGS% -o build\check_muon.exe src\check_muon.cu || exit /b 1

echo [9/9] check_ising
nvcc %FLAGS% -o build\check_ising.exe src\check_ising.cu || exit /b 1

echo.
echo built into build\
endlocal
