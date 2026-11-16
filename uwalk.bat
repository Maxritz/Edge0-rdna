@echo off
rem STRATA_LATENT_U_WALK=1: the alternative U kernel that keeps a row's ACTIVATION terms in registers and
rem walks the H output columns.  It measured 33.3 %% faster standalone but 15 %% slower in the engine when
rem the A stage streamed with the old 4-deep prefetch.  That prefetch is now 28 deep (fb1a943) and the A
rem stage issues 4x fewer, larger copies, so the cold-miss argument that killed it is worth re-testing.
rem Compare its [uprof] per-layer ms with the per-row kernel's 23.7 and its 8 tokens with the baseline.
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
set STRATA_DU_PROF=1
set STRATA_LATENT_U_WALK=1

strata.exe --pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats > C:\Users\rr\Desktop\Edge0-rdna\uwalk.log 2>&1
echo exit=%ERRORLEVEL%
endlocal
