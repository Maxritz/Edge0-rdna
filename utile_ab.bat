@echo off
rem The staged U tile (STRATA_LATENT_U_TILE=1): 256 output columns x 64 routed rows a block, both operands
rem in 4-byte aligned shared rows (64 codes + two half scales), the scale applied once per 16-value
rem sub-block, four __dp4a per sub-block on aligned words.  Compare [uprof] per-layer ms with the
rem per-row kernel's 23.7 and the ledger's gemm down (8,954 ms), and the 8 tokens with the baseline.
rem T1 = staged tile, T2 = staged tile again (a second pair, because one U figure is a single number)
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
set STRATA_DU_PROF=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats

echo ===== T1: staged tile =====
set STRATA_LATENT_U_TILE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\t1.log 2>&1
echo t1=%ERRORLEVEL%

echo ===== T2: staged tile (second pair) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\t2.log 2>&1
echo t2=%ERRORLEVEL%

echo ===== T3: per-row reference, same session =====
set STRATA_LATENT_U_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\t3.log 2>&1
echo t3=%ERRORLEVEL%
echo T_DONE
endlocal
