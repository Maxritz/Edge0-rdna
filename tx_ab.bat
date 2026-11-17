@echo off
rem Tile width, retested now that the kernel actually launches: per call qkv 3.17 (64x64) vs 2.84 (128x128),
rem gate 1.96 vs 1.81.  End to end, alternating, profiler on for the ledger and output checked every arm.
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
set STRATA_GEMM_DOT2=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== X64 =====
set STRATA_GEMM_DOT2_TILE=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tx64.log 2>&1
echo a=%ERRORLEVEL%
echo ===== X128 =====
set STRATA_GEMM_DOT2_TILE=128
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tx128.log 2>&1
echo b=%ERRORLEVEL%
echo ===== X64 again =====
set STRATA_GEMM_DOT2_TILE=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tx64b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== X128 again =====
set STRATA_GEMM_DOT2_TILE=128
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tx128b.log 2>&1
echo d=%ERRORLEVEL%
echo TX_DONE
endlocal