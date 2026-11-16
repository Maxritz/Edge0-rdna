@echo off
rem Two captured runs at 4K, back to back, so the per-node overhead can be separated from kernel time:
rem   trace_A_masked.log  - the workload as served (fence-fixed binary), per-stage GPU stamps on
rem   trace_C_nodes.log   - the same plus STRATA_DUMMY_NODES=64 (64 empty nodes per layer): A-C gives the
rem                         marginal cost of one graph node, which is the instrumentation/dispatch floor.
setlocal
set "ROOT=C:\Users\rr\Desktop\Edge0-rdna"

set STRATA_VERIFY_PROFILE=1
set STRATA_ROUTE_STATS=1
cmd /c "%ROOT%\bench4k.bat" > "%ROOT%\trace_A_baseline.log" 2>&1
echo A done

set STRATA_DUMMY_NODES=64
cmd /c "%ROOT%\bench4k.bat" > "%ROOT%\trace_C_nodes.log" 2>&1
echo C done

endlocal
