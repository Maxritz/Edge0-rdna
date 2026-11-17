@echo off
rem Global/DMA-contention test: does gemm gate/up follow the expert CACHE (i.e. the stream)?
rem 512 slots vs the default ~1900 vs 3000: if the stage is stream-bound the row must move with residency.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats
echo ===== cache 512 (4x less residency) =====
strata.exe %BASE% --expert-cache 512 --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\gc512.log 2>&1
echo a=%ERRORLEVEL%
echo ===== cache auto (~1900) =====
strata.exe %BASE% --expert-cache auto --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\gcauto.log 2>&1
echo b=%ERRORLEVEL%
echo ===== cache 3000 (more residency) =====
strata.exe %BASE% --expert-cache 3000 --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\gc3000.log 2>&1
echo c=%ERRORLEVEL%
echo ===== cache 512 again =====
strata.exe %BASE% --expert-cache 512 --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\gc512b.log 2>&1
echo d=%ERRORLEVEL%
echo GC_DONE
endlocal