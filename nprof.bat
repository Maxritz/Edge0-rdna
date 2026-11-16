@echo off
rem The fused A stage's own profile (prefill.cpp:5982, STRATA_LATENT_NPROF): calls, ms, entries, us/entry,
rem us/call, and the last call's group count / entry count / hbuf / cap / T / K.  The parity test measures
rem this kernel at 0.44-0.49 us/entry for the down half at >=32 entries per group; the engine's ledger
rem charges 9,192 ms to gemm down for ~2M entries, which would be ~4.6 us/entry.  This says which.
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build

set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_MTP_SIDE_STREAM=1
set STRATA_LATENT_NPROF=1
set STRATA_LAT_Q3K_INFO=1
set STRATA_PREFILL_TIMING=1

strata.exe --pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats > C:\Users\rr\Desktop\Edge0-rdna\nprof.log 2>&1
echo exit=%ERRORLEVEL%
endlocal
