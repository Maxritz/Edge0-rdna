@echo off
rem Decode is CPU-expert bound: 10.05 of 12.42 experts/layer go to the pool, wait-for-rings 59 of 96 ms.
rem More VRAM cache = fewer CPU experts.  Test the reserve, everything else fixed.
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build
set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q4.bin
set STRATA_LATENT_DENSE_FMT=
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_MTP_SIDE_STREAM=1
set STRATA_VERIFY_PROFILE=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 64
echo ===== reserve 512 =====
strata.exe %BASE% --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\dv512a.log 2>&1
echo a=%ERRORLEVEL%
echo ===== reserve 256 =====
strata.exe %BASE% --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\dv256a.log 2>&1
echo b=%ERRORLEVEL%
echo ===== reserve 512 again =====
strata.exe %BASE% --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\dv512b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== reserve 256 again =====
strata.exe %BASE% --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\dv256b.log 2>&1
echo d=%ERRORLEVEL%
echo DV_DONE
endlocal