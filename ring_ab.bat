@echo off
rem The copies are ring-starved, not link-bound: the probe says 28.0 GB/s (73.5 GB = 2.6 s of transfer)
rem but `wait copy` is 3.4 s and each copy occupies 41 us of DMA per 280 us of wall.  STRATA_STAGE_SLOTS
rem deepens the staging ring (32 slots = 37 MB today; the batch cap stage_n/2 follows it).
rem   R32 (default)   R64   R96   R32   R64
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats

echo ===== R32 =====
set STRATA_STAGE_SLOTS=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\r32.log 2>&1
echo r32=%ERRORLEVEL%
echo ===== R64 =====
set STRATA_STAGE_SLOTS=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\r64.log 2>&1
echo r64=%ERRORLEVEL%
echo ===== R96 =====
set STRATA_STAGE_SLOTS=96
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\r96.log 2>&1
echo r96=%ERRORLEVEL%
echo ===== R32 =====
set STRATA_STAGE_SLOTS=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\r32b.log 2>&1
echo r32b=%ERRORLEVEL%
echo ===== R64 =====
set STRATA_STAGE_SLOTS=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\r64b.log 2>&1
echo r64b=%ERRORLEVEL%
echo R_DONE
endlocal