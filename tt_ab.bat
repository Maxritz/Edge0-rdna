@echo off
rem dot2 GEMM tile width: 64x64 (16 acc a thread, 2:1 dot2:LDS) vs 128x128 (64 acc, 4:1).  rocBLAS's own number
rem for this stage is well established at 3,074-3,085 ms across many arms, so the A/B here is 64 vs 128 alternating.
rem Metric: the gdn ledger row.  Output recorded on every arm (differs from baseline by design; owner-approved).
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
echo ===== T64 =====
set STRATA_GEMM_DOT2_TILE=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tt64.log 2>&1
echo a=%ERRORLEVEL%
echo ===== T128 =====
set STRATA_GEMM_DOT2_TILE=128
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tt128.log 2>&1
echo b=%ERRORLEVEL%
echo ===== T64 again =====
set STRATA_GEMM_DOT2_TILE=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tt64b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== T128 again =====
set STRATA_GEMM_DOT2_TILE=128
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\tt128b.log 2>&1
echo d=%ERRORLEVEL%
echo TT_DONE
endlocal