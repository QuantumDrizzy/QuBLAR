@echo off
setlocal
cd /d "%~dp0"

rem Races and out-of-bounds accesses that no invariant above can see: a torn waveform bin
rem still looks like a plausible waveform, and a stack overrun in the traversal returns a
rem miss rather than crashing.
rem
rem Note the `call`. compute-sanitizer is itself a .bat, and cmd transfers control to a
rem batch file invoked without `call` and never comes back -- so the first invocation would
rem silently be the only one that ran, and the script would end without saying so.

call compute-sanitizer --tool memcheck  build\trace.exe       || exit /b 1
call compute-sanitizer --tool memcheck  build\check_lidar.exe || exit /b 1
call compute-sanitizer --tool racecheck build\check_lidar.exe || exit /b 1
call compute-sanitizer --tool memcheck  build\check_detector.exe || exit /b 1
call compute-sanitizer --tool racecheck build\check_detector.exe || exit /b 1
call compute-sanitizer --tool memcheck  build\check_transient.exe || exit /b 1
call compute-sanitizer --tool racecheck build\check_transient.exe || exit /b 1
call compute-sanitizer --tool memcheck build\check_muon.exe || exit /b 1
call compute-sanitizer --tool racecheck build\check_muon.exe || exit /b 1

rem check_external is deliberately absent: every GPU kernel it launches is
rem transient_trace, already sanitized above via check_transient, and the rest
rem of the binary is host-only code that compute-sanitizer does not see.
rem Re-running the full replica under the sanitizer would cost tens of minutes
rem and cover nothing new.

echo.
echo sanitizers clean
endlocal
