@echo off
rem Re-measure the fused A stage at the new 128-slot ring.  stage_batch = min(STRATA_LAT_STAGED_BATCH,
rem stage_n/2) and stage_n is now 128, so the cap is 64 - every earlier batch number (8/16/64 all 32.1 s)
rem was taken when stage_n was 8 and everything was silently clamped to 4, and the 16-vs-32 question has
rem never been asked at a ring that is not starved.
rem   B16 (default)   B32   B48   B16   B32
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats
echo ===== B16 =====
set STRATA_LAT_STAGED_BATCH=16
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b16.log 2>&1
echo b16=%ERRORLEVEL%
echo ===== B32 =====
set STRATA_LAT_STAGED_BATCH=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b32.log 2>&1
echo b32=%ERRORLEVEL%
echo ===== B48 =====
set STRATA_LAT_STAGED_BATCH=48
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b48.log 2>&1
echo b48=%ERRORLEVEL%
echo ===== B16 =====
set STRATA_LAT_STAGED_BATCH=16
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b16b.log 2>&1
echo b16b=%ERRORLEVEL%
echo ===== B32 =====
set STRATA_LAT_STAGED_BATCH=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b32b.log 2>&1
echo b32b=%ERRORLEVEL%
echo B_DONE
endlocal