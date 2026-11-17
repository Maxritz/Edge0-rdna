@echo off
rem The packed-fp16 (V_DOT2_F32_F16) dense GEMM vs the rocBLAS call it replaces, alternating in one binary via
rem STRATA_GEMM_DOT2.  Metric: the gdn ledger row (baseline 3,075-3,085 ms) and prefill tok/s.  The 32-token
rem output is recorded on every arm - it is EXPECTED to differ (different accumulation order; owner-approved).
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== D1 dot2 ON =====
set STRATA_GEMM_DOT2=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== D0 dot2 OFF (rocBLAS) =====
set STRATA_GEMM_DOT2=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt0.log 2>&1
echo b=%ERRORLEVEL%
echo ===== D1 again =====
set STRATA_GEMM_DOT2=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt1b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== D0 again =====
set STRATA_GEMM_DOT2=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt0b.log 2>&1
echo d=%ERRORLEVEL%
echo DD_DONE
endlocal