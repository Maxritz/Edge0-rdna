@echo off
rem LEAN: no pinning.  --prefill auto alone must land on 512 tokens at every cache size, by the engine's own
rem new geometry-based cap.  Three cache sizes, no --prefill argument, no STRATA_* overrides.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats
echo ===== L1 default cache =====
strata.exe %BASE% --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\ln1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== L2 reserve 640 =====
strata.exe %BASE% --vram-reserve-mib 640 > C:\Users\rr\Desktop\Edge0-rdna\ln2.log 2>&1
echo b=%ERRORLEVEL%
echo ===== L3 reserve 1024 =====
strata.exe %BASE% --vram-reserve-mib 1024 > C:\Users\rr\Desktop\Edge0-rdna\ln3.log 2>&1
echo c=%ERRORLEVEL%
echo LN_DONE
endlocal