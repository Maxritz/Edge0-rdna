@echo off
rem The GPU spends 24.7 ms of every 90 ms window spinning for the CPU pool, and the pool's 26.1 ms/round is
rem exactly 731 MB of expert weights streamed from DDR4 (30.5 GB/s).  The only thing that removes that traffic
rem outright is more experts resident in VRAM: 822 slots today = 6.7% of the 12,288 experts, 911 MiB.
rem The VRAM ledger leaves ~3.5 GB of the 11.2 GB budget unaccounted, so test whether it can actually be used.
rem   C1  reserve 256 MiB instead of 512 (what the auto sizer makes of it)
rem   C2  reserve 256 + an explicit 1400 slots (does the allocation hold?)
rem   C3  STRATA_EMBED_VRAM=0 as well (frees 644 MiB; only ever measured in the fault state)
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build

set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_MTP_SIDE_STREAM=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== C1: reserve 256, auto slots =====
set STRATA_EMBED_VRAM=1
strata.exe %BASE% --expert-cache auto --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\c1.log 2>&1
echo c1=%ERRORLEVEL%

echo ===== C2: reserve 256, 1400 slots =====
strata.exe %BASE% --expert-cache auto --vram-reserve-mib 256 --expert-cache-slots 1400 > C:\Users\rr\Desktop\Edge0-rdna\c2.log 2>&1
echo c2=%ERRORLEVEL%

echo ===== C3: reserve 256, embed off =====
set STRATA_EMBED_VRAM=0
strata.exe %BASE% --expert-cache auto --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\c3.log 2>&1
echo c3=%ERRORLEVEL%
set STRATA_EMBED_VRAM=1

echo CACHE_DONE
endlocal
