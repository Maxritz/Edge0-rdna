@echo off
rem The prompt path switches from per-expert GEMVs to MMQ (a tiled GEMM) at a chunk of stream_all_min() = 1024
rem tokens (prefill.cpp:252).  The shipped auto chunk on Windows HIP is 512, i.e. BELOW that floor, so the
rem tiled path is unreachable at the default by construction.  A bigger chunk also fixes the arithmetic
rem intensity: 512 tokens spread over ~13 experts per token gives each expert ~26 rows, and every chunk
rem re-streams the experts (63,357 blobs = 73.5 GB per prompt at 512).
rem   P1  --prefill 1024   (the MMQ floor)
rem   P2  --prefill 2048   (4x the rows per expert; the ring costs ~1.5 GiB of VRAM, so this may not fit)
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== P1: --prefill 1024 =====
strata.exe %BASE% --prefill 1024 > C:\Users\rr\Desktop\Edge0-rdna\q1.log 2>&1
echo q1=%ERRORLEVEL%

echo ===== P2: --prefill 2048 =====
strata.exe %BASE% --prefill 2048 > C:\Users\rr\Desktop\Edge0-rdna\q2.log 2>&1
echo q2=%ERRORLEVEL%

echo Q_DONE
endlocal
