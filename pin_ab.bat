@echo off
rem Decisive: does pinning the 26.66 GiB expert arena cost VRAM?  The engine's own comment says "host registration
rem takes no VRAM", but the measurement says only 4.43 GiB of 11.98 is free just after the page-lock while the
rem loaded weights account for 5.21 GiB of the 6.80 GiB that went missing.  STRATA_ARENA_PIN_GIB caps the pin
rem (0 = pin everything, the single-GPU default), so two caps against that default answer it directly.
rem   D1  cap 8 GiB      D2  cap 2 GiB
rem Watch: "VRAM after the arena page-lock", the resulting slot count, decode (the pool now reads less from
rem DDR4 if the cache is bigger) and prefill (pageable copies could cost the DMA path).
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== D1: arena pin capped at 8 GiB =====
set STRATA_ARENA_PIN_GIB=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\d5.log 2>&1
echo d1=%ERRORLEVEL%

echo ===== D2: arena pin capped at 2 GiB =====
set STRATA_ARENA_PIN_GIB=2
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\d6.log 2>&1
echo d2=%ERRORLEVEL%
set STRATA_ARENA_PIN_GIB=

echo D_DONE
endlocal
