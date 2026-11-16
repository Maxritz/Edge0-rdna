@echo off
rem The FFN-side hyper-connection read costs 20.3 ms/window (T4) for ~80 KB of matrices per layer, i.e. it is
rem launch-bound, not bandwidth-bound: ~12 tiny launches x 48 layers.  The engine already contains a single
rem persistent kernel for it (fused_gr_read_persist, src/kernels/cuda/fused_gr.cu:2830), bit-exact against the
rem default pair by hc_persist_parity, written for 32-lane waves and gated off on gfx11/gfx12 (no s_memrealtime).
rem gfx1031 HAS s_memrealtime, so it should engage here.  It is opt-in and off by default.
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
set STRATA_VERIFY_PROFILE=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== H1: STRATA_HC_PERSIST=1 (one persistent HC read) =====
set STRATA_HC_PERSIST=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\h1.log 2>&1
echo h1=%ERRORLEVEL%
set STRATA_HC_PERSIST=

echo H_DONE
endlocal
