@echo off
rem The A stage's launch count is set by stage_batch = min(STRATA_LAT_STAGED_BATCH, stage_n/2), and stage_n
rem (stage_slots) is already 32, so the ring is NOT the limit - the default batch of 4 is.  Measured shape of
rem one launch (STRATA_LATENT_NPROF): 4 groups, 64 entries, 238 us -> 3.08 us/entry, i.e. ~30,700 launches
rem for a 4K prompt.  Every 4x on the batch quarters that count.
rem   B0 default 4   B1 8   B2 16 (the cap)   B3 32 (capped to 16, proves the cap)
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
set STRATA_LATENT_NPROF=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats

echo ===== B1: batch 8 =====
set STRATA_LAT_STAGED_BATCH=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b1.log 2>&1
echo b1=%ERRORLEVEL%

echo ===== B2: batch 16 =====
set STRATA_LAT_STAGED_BATCH=16
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b2.log 2>&1
echo b2=%ERRORLEVEL%

echo ===== B3: batch 64 (should cap at stage_n/2) =====
set STRATA_LAT_STAGED_BATCH=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\b3.log 2>&1
echo b3=%ERRORLEVEL%
set STRATA_LAT_STAGED_BATCH=

echo B_DONE
endlocal
