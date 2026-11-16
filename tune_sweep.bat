@echo off
rem Tuning sweep at 4K. Each run is one change against trace_A_baseline.log (23.71 tok/s decode,
rem 93.03 tok/s prefill, 95.99 ms T=4 window). One variable at a time.
setlocal
set "ROOT=C:\Users\rr\Desktop\Edge0-rdna"

rem ---- P1: a chunk of 1024 instead of auto's 512. Two effects at once, both wanted and both named:
rem   (a) the deep streamed ring engages at stream_all_min()==1024, replacing the 8-slot staged walk;
rem   (b) 4 passes over the expert set instead of 8.
cmd /c "%ROOT%\bench4k.bat --prefill 1024" > "%ROOT%\tune_P1_chunk1024.log" 2>&1
echo P1 done

rem ---- P2: P1 plus the two documented copy-event coalescing knobs (only live when the streamed ring runs)
set STRATA_PREFILL_COPY_EVENT_EVERY=32
set STRATA_PREFILL_COPY_WAIT_DEDUPE=1
cmd /c "%ROOT%\bench4k.bat --prefill 1024" > "%ROOT%\tune_P2_coalesce.log" 2>&1
set STRATA_PREFILL_COPY_EVENT_EVERY=
set STRATA_PREFILL_COPY_WAIT_DEDUPE=
echo P2 done

rem ---- D1: the decode-side launch levers. FUSE_HITS folds moe_hit_add into the combine (gfx12-only by
rem default, so OFF on gfx1031); HC_PERSIST uses the persistent fused hyper-connection read.
set STRATA_FUSE_HITS=1
set STRATA_HC_PERSIST=1
cmd /c "%ROOT%\bench4k.bat" > "%ROOT%\tune_D1_launch.log" 2>&1
set STRATA_FUSE_HITS=
set STRATA_HC_PERSIST=
echo D1 done

endlocal
