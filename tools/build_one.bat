@echo off
rem Build one check from src\<name>.cu into build\<name>.exe, the way build.bat does.
rem Exit code is nvcc's, so a failed build stops tools\ledger_batch.py.
rem This file must keep CRLF line endings (see build.bat).
setlocal
cd /d "%~dp0.."
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not exist build mkdir build
nvcc -O3 -arch=sm_120 -std=c++17 -lineinfo -o build\%1.exe src\%1.cu
exit /b %ERRORLEVEL%
