@echo off
rem The latent A stage has THREE implementations and the prompt runs the third:
rem   1. the native per-expert grouped walk          (STRATA_LAT_FUSED_Q3K=0, no MMQ)
rem   2. the MMQ walk, one product per MMQ_GROUP      (STRATA_LAT_FUSED_Q3K=0 + STRATA_LATENT_MMQ=1)
rem   3. lat_q3k_fused, experts read in place         (default for this pack)
rem lat_q3k_fused is chosen BEFORE lat_mmq (prefill.cpp:5678 vs :5699), so the tiled MMQ product has never
rem run on this pack.  The fused kernels are the two that dominate the ledger (gemm gate/up 8.4 s + gemm
rem down 9.2 s of a 32.3 s prompt), so both alternatives are worth a measured look.
rem   M1  MMQ        M2  native grouped
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build

set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_MTP_SIDE_STREAM=1
set STRATA_PREFILL_TIMING=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 16 --stats

echo ===== M1: MMQ A stage =====
set STRATA_LAT_FUSED_Q3K=0
set STRATA_LATENT_MMQ=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\m1.log 2>&1
echo m1=%ERRORLEVEL%

echo ===== M2: native grouped A stage =====
set STRATA_LATENT_MMQ=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\m2.log 2>&1
echo m2=%ERRORLEVEL%
set STRATA_LAT_FUSED_Q3K=
set STRATA_LATENT_MMQ=

echo M_DONE
endlocal
