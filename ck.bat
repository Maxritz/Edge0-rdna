@echo off
rem Expert-outer, tested by configuration: with the ring now 512, do bigger chunks finally stream FEWER blobs?
rem Expert-outer predicts 62,939 -> ~24,000 fetches at one pass per layer.
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
set STRATA_PREFILL_TIMING=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 8192 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== chunk auto (512) =====
strata.exe %BASE% --prefill auto > C:\Users\rr\Desktop\Edge0-rdna\ck_auto.log 2>&1
echo a=%ERRORLEVEL%
echo ===== chunk 1024 =====
strata.exe %BASE% --prefill 1024 > C:\Users\rr\Desktop\Edge0-rdna\ck1024.log 2>&1
echo b=%ERRORLEVEL%
echo ===== chunk 2048 =====
strata.exe %BASE% --prefill 2048 > C:\Users\rr\Desktop\Edge0-rdna\ck2048.log 2>&1
echo c=%ERRORLEVEL%
echo ===== chunk 4096 =====
strata.exe %BASE% --prefill 4096 > C:\Users\rr\Desktop\Edge0-rdna\ck4096.log 2>&1
echo d=%ERRORLEVEL%
echo CK_DONE
endlocal