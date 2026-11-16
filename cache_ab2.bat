@echo off
rem Two configurations, two runs each, alternating, so the comparison is not one noisy sample:
rem   A  --vram-reserve-mib 512   -> 822 slots   (the current harness setting)
rem   B  --vram-reserve-mib 256   -> 1057 slots  (the most the sizer will give with the embedding in VRAM)
rem Everything else identical; prefill is the same prompt so both columns move together.
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== A1 reserve 512 =====
strata.exe %BASE% --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\ca1.log 2>&1
echo a1=%ERRORLEVEL%
echo ===== B1 reserve 256 =====
strata.exe %BASE% --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\cb1.log 2>&1
echo b1=%ERRORLEVEL%
echo ===== A2 reserve 512 =====
strata.exe %BASE% --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\ca2.log 2>&1
echo a2=%ERRORLEVEL%
echo ===== B2 reserve 256 =====
strata.exe %BASE% --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\cb2.log 2>&1
echo b2=%ERRORLEVEL%
echo CAB_DONE
endlocal
