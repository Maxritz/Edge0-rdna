@echo off
rem DECISIVE: is the chunk the cause, or the cache size?  Both arms at the same cache (reserve 1024 =
rem ~557-570 slots), one with --prefill auto (picks 1024) and one pinned at 512.  The earlier
rem "1024 = 203 tok/s" comparison was against a 796-slot run, so it conflated the two.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats --vram-reserve-mib 1024
echo ===== auto (picks 1024) =====
strata.exe %BASE% --prefill auto > C:\Users\rr\Desktop\Edge0-rdna\cs_auto.log 2>&1
echo a=%ERRORLEVEL%
echo ===== pinned 512 =====
strata.exe %BASE% --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\cs_512.log 2>&1
echo b=%ERRORLEVEL%
echo ===== auto again =====
strata.exe %BASE% --prefill auto > C:\Users\rr\Desktop\Edge0-rdna\cs_auto2.log 2>&1
echo c=%ERRORLEVEL%
echo ===== pinned 512 again =====
strata.exe %BASE% --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\cs_512b.log 2>&1
echo d=%ERRORLEVEL%
echo CS_DONE
endlocal