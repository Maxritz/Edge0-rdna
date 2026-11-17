@echo off
rem With Q4 the cache has 1910 slots; chunk 512 + ring 128 + 128 uses 768.  Does a bigger chunk cash the rest in?
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
set STRATA_STAGE_SLOTS=128
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== chunk 1024 ring 128 =====
strata.exe %BASE% --prefill 1024 > C:\Users\rr\Desktop\Edge0-rdna\c1ka.log 2>&1
echo a=%ERRORLEVEL%
echo ===== chunk 512 ring 128 (control) =====
strata.exe %BASE% --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\c512a.log 2>&1
echo b=%ERRORLEVEL%
echo ===== chunk 2048 ring 128 =====
strata.exe %BASE% --prefill 2048 > C:\Users\rr\Desktop\Edge0-rdna\c2ka.log 2>&1
echo c=%ERRORLEVEL%
echo CHUNK_DONE
endlocal