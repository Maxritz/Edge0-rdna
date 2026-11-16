@echo off
rem D (the "gather" phase, 3,599 ms = 17.6 % of the prompt) on the staged tile: no groups at all (every
rem token takes the whole [G*ELL, N] matrix), so the tile runs 8 full 64-row tiles at T=512 - no padding.
rem Gate: the 8 tokens must match the iq_mmvq arm, and the ledger's `gather` row (3,599) is the metric.
rem   D0 iq_mmvq (shipped)   D1 staged tile   D2 staged tile (second pair)
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
set STRATA_D_PROF=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats

echo ===== D0: iq_mmvq =====
set STRATA_LATENT_D_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\d0.log 2>&1
echo d0=%ERRORLEVEL%

echo ===== D1: staged tile =====
set STRATA_LATENT_D_TILE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\d1.log 2>&1
echo d1=%ERRORLEVEL%

echo ===== D2: staged tile (second pair) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\d2.log 2>&1
echo d2=%ERRORLEVEL%
echo D_DONE
endlocal
