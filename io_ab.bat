@echo off
rem The copy-issue bucket is 8,132 ms of host time, mostly Stager::wait.  STRATA_IO_THREADS sizes the issuing
rem pool (expert_source io_workers_, direct_file).  The AMD reference config uses 32.  Metric: prefill tok/s,
rem the host-window buckets and the prefill ledger; gate: 32-token output.
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
set STRATA_PREFILL_HOSTPROF=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== I8 (default) =====
set STRATA_IO_THREADS=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\io8.log 2>&1
echo a=%ERRORLEVEL%
echo ===== I32 =====
set STRATA_IO_THREADS=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\io32.log 2>&1
echo b=%ERRORLEVEL%
echo ===== I8 again =====
set STRATA_IO_THREADS=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\io8b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== I32 again =====
set STRATA_IO_THREADS=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\io32b.log 2>&1
echo d=%ERRORLEVEL%
echo IO_DONE
endlocal