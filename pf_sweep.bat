@echo off
rem Prefill is the 1/6 number. Four no-rebuild levers, one variable each, plus the trim probes.
rem  A: chunk 1024 (unlocks the deep streamed ring at stream_all_min() and halves the passes over the expert set)
rem  B: chunk 2048 (quarter the passes)
rem  C: chunk 1024 + the two documented copy-event coalescing knobs (one WDDM event/wait per N experts)
rem  D: trim probe: COUNT ONLY. How much would a rank-R cut actually drop, at zero risk.
setlocal
set "ROOT=C:\Users\rr\Desktop\Edge0-rdna"

cmd /c "%ROOT%\bench4k.bat --prefill 1024" > "%ROOT%\pf_A_1024.log" 2>&1
echo A done

cmd /c "%ROOT%\bench4k.bat --prefill 2048" > "%ROOT%\pf_B_2048.log" 2>&1
echo B done

set STRATA_PREFILL_COPY_EVENT_EVERY=8
set STRATA_PREFILL_COPY_WAIT_DEDUPE=1
cmd /c "%ROOT%\bench4k.bat --prefill 1024" > "%ROOT%\pf_C_coalesce.log" 2>&1
set STRATA_PREFILL_COPY_EVENT_EVERY=
set STRATA_PREFILL_COPY_WAIT_DEDUPE=
echo C done

rem Count-only probes: PREFILL_PRUNE_PROBE says how many streamed blobs a rank cut would drop,
rem PRUNE_STATS says the same for the token side's (row, slot) pairs plus the router mass lost.
set STRATA_PREFILL_PRUNE_PROBE=8
set STRATA_PRUNE_STATS=8
cmd /c "%ROOT%\bench4k.bat" > "%ROOT%\pf_D_trimprobe.log" 2>&1
echo D done

endlocal
