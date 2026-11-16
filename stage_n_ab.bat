@echo off
rem stage_n was hard-coded to STAGE (8) for a streamed walk, so stage_batch = min(batch, stage_n/2) capped the
rem fused A stage at FOUR groups and every earlier batch request (8/16/64) was silently clamped to 4 - which is
rem exactly why all three measured the same 32.1 s.  stage_n now follows stage_slots() = 32 (the buffers already
rem existed), so the batch can finally be exercised:
rem   N1 batch 4  (what the engine has been running all along)
rem   N2 batch 16 (stage_n/2, the cap)
rem Watch the ledger's gemm gate/up (8,420 / 8,424) and gemm down (9,718 / 9,192), and the profile's us/call.
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

echo ===== N1: batch 4 =====
set STRATA_LAT_STAGED_BATCH=4
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\n1.log 2>&1
echo n1=%ERRORLEVEL%

echo ===== N2: batch 16 =====
set STRATA_LAT_STAGED_BATCH=16
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\n2.log 2>&1
echo n2=%ERRORLEVEL%
set STRATA_LAT_STAGED_BATCH=

echo N_DONE
endlocal
