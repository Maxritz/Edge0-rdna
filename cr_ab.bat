@echo off
rem Ring depth AT A FIXED CHUNK, which the auto-chunk A/B could not isolate.  prefill.cpp:413 records that 96
rem slots (246.3 tok/s) beats 64 (242.5) at a fixed chunk, so lowering the ring default may cost more than the
rem chunk it buys.  This measures the ring's own curve at chunk 512 / 721 slots, so the shipping decision is
rem either "ring default 80" or "relax the +128 lend-headroom test and keep 96".
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
echo ===== C512 R96 =====
set STRATA_STAGE_SLOTS=96
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cr96.log 2>&1
echo a=%ERRORLEVEL%
echo ===== C512 R80 =====
set STRATA_STAGE_SLOTS=80
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cr80.log 2>&1
echo b=%ERRORLEVEL%
echo ===== C512 R112 =====
set STRATA_STAGE_SLOTS=112
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cr112.log 2>&1
echo c=%ERRORLEVEL%
echo ===== C512 R96 again =====
set STRATA_STAGE_SLOTS=96
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cr96b.log 2>&1
echo d=%ERRORLEVEL%
echo CR_DONE
endlocal