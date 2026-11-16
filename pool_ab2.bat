@echo off
rem Second pair for --pool-tasks 36.  The metric that matters is the pool's own bandwidth over its rows
rem phases (averaged over 48 layers x 13 rounds), not a 32-token decode figure:
rem   baseline  30.4 GB/s, pool 27.63 ms/round, decode 24.87
rem   tasks 36  31.7 GB/s, pool 26.71 ms/round, decode 25.90
rem 9 workers was 26.5 GB/s / 23.27, i.e. clearly worse (SMT contention), so workers stay at 5.
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== X: baseline =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w3.log 2>&1
echo w3=%ERRORLEVEL%
echo ===== Y: tasks 36 =====
strata.exe %BASE% --pool-tasks 36 > C:\Users\rr\Desktop\Edge0-rdna\w4.log 2>&1
echo w4=%ERRORLEVEL%
echo ===== X: baseline =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w5.log 2>&1
echo w5=%ERRORLEVEL%
echo ===== Y: tasks 36 =====
strata.exe %BASE% --pool-tasks 36 > C:\Users\rr\Desktop\Edge0-rdna\w6.log 2>&1
echo w6=%ERRORLEVEL%
echo WP_DONE
endlocal
