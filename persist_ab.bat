@echo off
rem Honest A/B for the HC persist default I shipped: two pairs, alternating, and the comparison metric is the
rem averaged T4 window (PROF-T, 7-8 windows per run) rather than one 32-token decode figure, because a single
rem decode number here varies by ~9% between identical runs (ca1 25.02 / ca2 25.55 / cb1 26.84 / cb2 24.56).
rem   X  STRATA_HC_PERSIST=0   (the default read)
rem   Y  STRATA_HC_PERSIST=1   (the persistent read, now the gfx10 default)
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
set STRATA_VERIFY_PROFILE=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== X1 persist OFF =====
set STRATA_HC_PERSIST=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\px1.log 2>&1
echo x1=%ERRORLEVEL%
echo ===== Y1 persist ON =====
set STRATA_HC_PERSIST=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\py1.log 2>&1
echo y1=%ERRORLEVEL%
echo ===== X2 persist OFF =====
set STRATA_HC_PERSIST=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\px2.log 2>&1
echo x2=%ERRORLEVEL%
echo ===== Y2 persist ON =====
set STRATA_HC_PERSIST=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\py2.log 2>&1
echo y2=%ERRORLEVEL%
set STRATA_HC_PERSIST=
echo PAB_DONE
endlocal
