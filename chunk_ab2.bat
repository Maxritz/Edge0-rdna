@echo off
rem With the U projection fixed (2.7 ms a layer) the A stage is the largest item (5,584 ms of 20,400).
rem Its tile is 64 routed rows but STRATA_LATENT_NPROF says a group carries ~25 entries at a 512-token
rem chunk (nb=16 ent=396 -> 24.75), so the tile is 39 %% full.  A 1024-token chunk would double the
rem entries per group (~50, 78 %% full) at the cost of more streamed blobs - and the earlier chunk-1024
rem test (31,773 vs 32,280) predates the stage_n/batch fix, so it measured a clamped walk.  Re-test both.
rem   C0 chunk auto (512)   C1 chunk 1024
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
set STRATA_PREFILL_TIMING=1
set STRATA_LATENT_NPROF=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats

echo ===== C0: chunk auto (512) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\c0.log 2>&1
echo c0=%ERRORLEVEL%

echo ===== C1: chunk 1024 =====
strata.exe %BASE% --prefill 1024 > C:\Users\rr\Desktop\Edge0-rdna\c1.log 2>&1
echo c1=%ERRORLEVEL%
echo C_DONE
endlocal
