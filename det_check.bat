@echo off
rem Is greedy-token equality a valid A/B gate on this engine?  Two runs of the SAME binary, identical flags,
rem then one run with the staged batch raised.  If A1 != A2 with nothing changed, the gate is invalid and any
rem verdict drawn from tokens is void.
setlocal
set "ROOT=C:\Users\rr\Desktop\Edge0-rdna"

cmd /c "%ROOT%\bench4k.bat" > "%ROOT%\det_A1.log" 2>&1
cmd /c "%ROOT%\bench4k.bat" > "%ROOT%\det_A2.log" 2>&1
set STRATA_LAT_STAGED_BATCH=8
cmd /c "%ROOT%\bench4k.bat" > "%ROOT%\det_B8.log" 2>&1
set STRATA_LAT_STAGED_BATCH=

endlocal
