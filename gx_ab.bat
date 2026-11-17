@echo off
rem The dot2 GEMM now actually launches (0.59: the launch was inside a device-arch guard the host pass never sees,
rem which is why three earlier "measurements" compared rocBLAS with itself).  Per-call it is 3.20 ms on qkv
rem against rocBLAS's 6.94.  End-to-end A/B, alternating, chunk 512.
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
set STRATA_GEMM_DOT2_TILE=64
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== A1 dot2 ON =====
set STRATA_GEMM_DOT2=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\gx1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== A0 rocBLAS =====
set STRATA_GEMM_DOT2=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\gx0.log 2>&1
echo b=%ERRORLEVEL%
echo ===== A1 again =====
set STRATA_GEMM_DOT2=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\gx1b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== A0 again =====
set STRATA_GEMM_DOT2=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\gx0b.log 2>&1
echo d=%ERRORLEVEL%
echo GX_DONE
endlocal