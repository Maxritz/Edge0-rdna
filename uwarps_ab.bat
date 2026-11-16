@echo off
rem Decode's U is bound by block count: 320 blocks a routed row (8 columns each) x 10 rows x 48 layers
rem = ~154k blocks a round at ~86 ns = the 13.2 ms in "u: gemv".  STRATA_LATENT_U_WARPS=N gives a block
rem N warps / N columns instead of 8 - bit-identical arithmetic, 4x fewer blocks at N=32.  The round
rem table's "u: gemv" and decode tok/s are the metrics; `output` is the identity gate.
rem   V8 (default)   V32   V8   V32   V16
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
set STRATA_VERIFY_U_PROF=1
set STRATA_VERIFY_PROFILE=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== V8 =====
set STRATA_LATENT_U_WARPS=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\y8.log 2>&1
echo y8=%ERRORLEVEL%
echo ===== V32 =====
set STRATA_LATENT_U_WARPS=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\y32.log 2>&1
echo y32=%ERRORLEVEL%
echo ===== V8 =====
set STRATA_LATENT_U_WARPS=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\y8b.log 2>&1
echo y8b=%ERRORLEVEL%
echo ===== V32 =====
set STRATA_LATENT_U_WARPS=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\y32b.log 2>&1
echo y32b=%ERRORLEVEL%
echo Y_DONE
endlocal
