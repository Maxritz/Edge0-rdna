@echo off
rem The 3/4-ring hysteresis changed nothing, so the ring is not the binding test.  The scan's other condition is
rem `k + 128 <= xcache.slots()` with k = chunk + ring: at 721 slots chunk 512 + ring 96 = 608, and 608 + 128 = 736
rem > 721, so 512 is rejected on LEND HEADROOM and auto falls to 256.  With ring 80, 592 + 128 = 720 <= 721.
rem Test with the existing binary via STRATA_STAGE_SLOTS - no rebuild.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== R96 (shipped ring, reference) =====
set STRATA_STAGE_SLOTS=96
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rs96.log 2>&1
echo a=%ERRORLEVEL%
echo ===== R80 =====
set STRATA_STAGE_SLOTS=80
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rs80.log 2>&1
echo b=%ERRORLEVEL%
echo ===== R64 =====
set STRATA_STAGE_SLOTS=64
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rs64.log 2>&1
echo c=%ERRORLEVEL%
echo ===== R80 again =====
set STRATA_STAGE_SLOTS=80
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rs80b.log 2>&1
echo d=%ERRORLEVEL%
echo RS_DONE
endlocal