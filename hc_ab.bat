@echo off
rem hc1 norm is 26.76 ms/round (22.9 % of decode) and is already a slot the HC kernel stamps itself, so the
rem question is which HC mode produces it.  Three arms: the shipped default (persistent read, staged), the
rem pre-persistent path (STRATA_HC_PERSIST=0) and the older staged variant (STRATA_HC_SPLIT=1).
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== H1 default (persist) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hc1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== H2 persist=0 =====
set STRATA_HC_PERSIST=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hc2.log 2>&1
echo b=%ERRORLEVEL%
echo ===== H3 split=1 =====
set STRATA_HC_PERSIST=
set STRATA_HC_SPLIT=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hc3.log 2>&1
echo c=%ERRORLEVEL%
echo HC_DONE
endlocal