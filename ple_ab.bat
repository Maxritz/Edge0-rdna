@echo off
rem Decode now spends 40.5 ms/token of which only 20.8 is the GPU window.  The rest is the PLE
rem n-gram table: 69,728 SSD reads at p50 5.7 ms, 0.1% row-cache hits, 294.7 MB moved - for a table
rem that is ~4.4 GB and would sit in this box's 48 GB of RAM.
rem
rem The harness pinned --ple-io direct, which also disables the engine's auto-mmap heuristic
rem (generate.cpp:4959 auto_mmap = !ple_io_explicit).  Three modes, same everything else:
rem   P1  --ple-io ram   : mmap + lock (the pages stay resident in RAM)
rem   P2  --ple-io mmap  : mmap (page cache, no lock)
rem   P3  no --ple-io    : the engine's own choice (auto-mmap when direct measures slow)
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --spec 4 --spec-min-p 0.5 --stats

echo ===== P1: --ple-io ram =====
strata.exe %BASE% --ple-io ram > C:\Users\rr\Desktop\Edge0-rdna\p1.log 2>&1
echo p1=%ERRORLEVEL%

echo ===== P2: --ple-io mmap =====
strata.exe %BASE% --ple-io mmap > C:\Users\rr\Desktop\Edge0-rdna\p2.log 2>&1
echo p2=%ERRORLEVEL%

echo ===== P3: no --ple-io (engine's own choice) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\p3.log 2>&1
echo p3=%ERRORLEVEL%

echo PLE_DONE
endlocal
