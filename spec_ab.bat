@echo off
rem The decode D stage reads 52 MB a layer per WINDOW (G*ELL x N of Q8_0), so its cost per token is
rem 52*48/T MB - it halves when the speculative window doubles.  That makes the MTP depth a decode knob
rem that pays directly on the 21.46 ms the last commit attributed: --spec N is the draft window.
rem Speculative decoding is lossless, so this is a policy knob, not an arithmetic change.
rem   S4 (what every harness has used)   S6   S8   S4   S8
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== S4 =====
strata.exe %BASE% --spec 4 > C:\Users\rr\Desktop\Edge0-rdna\p4.log 2>&1
echo p4=%ERRORLEVEL%
echo ===== S6 =====
strata.exe %BASE% --spec 6 > C:\Users\rr\Desktop\Edge0-rdna\p6.log 2>&1
echo p6=%ERRORLEVEL%
echo ===== S8 =====
strata.exe %BASE% --spec 8 > C:\Users\rr\Desktop\Edge0-rdna\p8.log 2>&1
echo p8=%ERRORLEVEL%
echo ===== S4 =====
strata.exe %BASE% --spec 4 > C:\Users\rr\Desktop\Edge0-rdna\p4b.log 2>&1
echo p4b=%ERRORLEVEL%
echo ===== S8 =====
strata.exe %BASE% --spec 8 > C:\Users\rr\Desktop\Edge0-rdna\p8b.log 2>&1
echo p8b=%ERRORLEVEL%
echo P_DONE
endlocal
