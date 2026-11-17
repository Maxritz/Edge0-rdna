@echo off
rem gdn (3,086 ms) is arithmetic-bound: the four dense projections per layer are the f16 GEMM, and on HIP
rem Gemm::f16 reaches strata_pf_gemm_f16 ONLY with STRATA_PF_GEMM=1 (it is opt-in).  The gfx11 PADDED variant is
rem unavailable on gfx1031, but the non-padded strata_pf_gemm_f16 may not be.  Also STRATA_WMMA_GEMM=1, which is
rem RDNA3+ and should fall through harmlessly.  Metric: the gdn ledger row; gate: 32-token output.
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
echo ===== G0 default =====
set STRATA_PF_GEMM=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pg0.log 2>&1
echo a=%ERRORLEVEL%
echo ===== G1 STRATA_PF_GEMM=1 =====
set STRATA_PF_GEMM=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pg1.log 2>&1
echo b=%ERRORLEVEL%
echo ===== G2 PF_GEMM=1 + WMMA_GEMM=1 =====
set STRATA_WMMA_GEMM=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pg2.log 2>&1
echo c=%ERRORLEVEL%
echo ===== G0 again =====
set STRATA_WMMA_GEMM=
set STRATA_PF_GEMM=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pg0b.log 2>&1
echo d=%ERRORLEVEL%
echo PG_DONE
endlocal