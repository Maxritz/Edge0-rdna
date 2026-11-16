@echo off
rem The pinned pool reaches 30.5-32.2 GB/s while the engine's own expert-row bench (kq_cpu_bench, which
rem cannot pin on Windows - pin_to is a no-op there) reaches 37.2-37.6 GB/s with 8 threads on the same
rem box.  STRATA_POOL_NO_PIN=1 lets the OS place the pool's threads instead of the plan's core set.
rem Alternating pairs, 32 tokens; the metric is the pool's own GB/s and decode tok/s.
rem   N0 pinned (default)   N1 unpinned   N0   N1
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

echo ===== N0: pinned =====
set STRATA_POOL_NO_PIN=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\n0.log 2>&1
echo n0=%ERRORLEVEL%
echo ===== N1: unpinned =====
set STRATA_POOL_NO_PIN=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\n1.log 2>&1
echo n1=%ERRORLEVEL%
echo ===== N0: pinned =====
set STRATA_POOL_NO_PIN=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\n0b.log 2>&1
echo n0b=%ERRORLEVEL%
echo ===== N1: unpinned =====
set STRATA_POOL_NO_PIN=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\n1b.log 2>&1
echo n1b=%ERRORLEVEL%
echo N_DONE
endlocal
