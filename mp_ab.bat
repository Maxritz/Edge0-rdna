@echo off
rem The main compute stream on the high-priority hardware queue (AMD HQD priority) vs default priority.
rem Alternating pairs, output checked every arm.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== P1 high-priority main queue =====
set STRATA_MAIN_PRIO=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\mp1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== P0 default priority =====
set STRATA_MAIN_PRIO=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\mp0.log 2>&1
echo b=%ERRORLEVEL%
echo ===== P1 again =====
set STRATA_MAIN_PRIO=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\mp1b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== P0 again =====
set STRATA_MAIN_PRIO=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\mp0b.log 2>&1
echo d=%ERRORLEVEL%
echo MP_DONE
endlocal