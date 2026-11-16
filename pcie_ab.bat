@echo off
rem decode's 24.7 ms/round pcie/uWait is the CPU pool moving its share of the round's expert bytes at
rem ~30 GB/s.  The pool is at the machine's ceiling (4/5/6/7 threads all measure 29-32 GB/s), so the
rem remaining lever is HOW MUCH goes to the pool at all: --pcie-frac is the share of misses read over
rem PCIe instead, and the pool and the link run in PARALLEL - so the optimum is where both finish
rem together (f such that f/P = (1 - f)/30).  Every harness in this tree had been forcing 0.1, while the
rem engine's own default for a native pack is 0.55; `auto` refines it from the measured link.
rem   F1 = 0.1 (what every measurement so far used)   F2 = 0.25   F3 = 0.55 (the default)   F4 = auto
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --gpu 0 --max-new 32 --stats

echo ===== F1: pcie-frac 0.1 =====
strata.exe %BASE% --pcie-frac 0.1 > C:\Users\rr\Desktop\Edge0-rdna\f1.log 2>&1
echo f1=%ERRORLEVEL%
echo ===== F3: pcie-frac 0.55 =====
strata.exe %BASE% --pcie-frac 0.55 > C:\Users\rr\Desktop\Edge0-rdna\f3.log 2>&1
echo f3=%ERRORLEVEL%
echo ===== F1: pcie-frac 0.1 =====
strata.exe %BASE% --pcie-frac 0.1 > C:\Users\rr\Desktop\Edge0-rdna\f1b.log 2>&1
echo f1b=%ERRORLEVEL%
echo ===== F3: pcie-frac 0.55 =====
strata.exe %BASE% --pcie-frac 0.55 > C:\Users\rr\Desktop\Edge0-rdna\f3b.log 2>&1
echo f3b=%ERRORLEVEL%
echo ===== F4: pcie-frac auto =====
strata.exe %BASE% --pcie-frac auto > C:\Users\rr\Desktop\Edge0-rdna\f4.log 2>&1
echo f4=%ERRORLEVEL%
echo F_DONE
endlocal
