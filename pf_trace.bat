@echo off
rem The prefill breadcrumb trace, two chunk sizes, same prompt, same everything else.
rem STRATA_PREFILL_TIMING=1 gives the per-phase dev+host table; STRATA_LATENT_NPROF=1 times the A stage
rem function itself (lat_q3k_fused) at us/entry, which the phase table cannot see inside.
setlocal
set "ROOT=C:\Users\rr\Desktop\Edge0-rdna"
set STRATA_PREFILL_TIMING=1
set STRATA_LATENT_NPROF=1

cmd /c "%ROOT%\bench4k.bat --dump-routing %ROOT%\routing_512.bin" > "%ROOT%\tr_512.log" 2>&1
echo T512 done

cmd /c "%ROOT%\bench4k.bat --prefill 2048" > "%ROOT%\tr_2048.log" 2>&1
echo T2048 done

endlocal
