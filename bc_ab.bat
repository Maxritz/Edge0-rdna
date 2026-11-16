@echo off
rem BASELINE RECONCILIATION.  The recorded "249.0/248.5 prefill" was taken at 796-803 cache slots
rem (1.89-1.90 GiB free); today the box offers 1.82-1.83 GiB and --expert-cache auto lands on 735-742.
rem A1 = the recorded canonical harness UNMODIFIED (--prefill auto, --expert-cache auto).
rem A2 = the same with the cache pinned to the recorded working set (800 slots) and chunk 512.
rem If A2 lands near 249 the whole "regression" is free-VRAM on the box, not a code change.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== A1 canonical unmodified (auto/auto) =====
strata.exe %BASE% --expert-cache auto --prefill auto > C:\Users\rr\Desktop\Edge0-rdna\bc1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== A2 cache pinned to the recorded 800 slots, chunk 512 =====
strata.exe %BASE% --expert-cache 800 --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\bc2.log 2>&1
echo b=%ERRORLEVEL%
echo ===== A2 again =====
strata.exe %BASE% --expert-cache 800 --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\bc3.log 2>&1
echo c=%ERRORLEVEL%
echo ===== A1 again =====
strata.exe %BASE% --expert-cache auto --prefill auto > C:\Users\rr\Desktop\Edge0-rdna\bc4.log 2>&1
echo d=%ERRORLEVEL%
echo BC_DONE
endlocal