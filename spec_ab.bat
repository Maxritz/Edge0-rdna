@echo off
rem Decode's window mix is the cheapest real lever, and it is pure scheduling:
rem   1 token  costs 59.72 ms/token (T1)
rem   4 tokens cost 93.22 ms = 23.30 ms/token (T4)
rem 8 of 13 rounds hit T4; the other 5 (T1 x3, T2, T3) are what hold the blend at 23.56 tok/s.
rem --spec-min-p truncates a draft window when a drafted token's probability is below the threshold,
rem so it directly decides how many rounds get 4 tokens.  Lower it and measure: prefill is identical
rem in every run (same prompt), so the decode figure is the only thing under test.
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
set STRATA_VERIFY_PROFILE=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== S1: spec 4, min-p 0.20 =====
strata.exe %BASE% --spec 4 --spec-min-p 0.2 > C:\Users\rr\Desktop\Edge0-rdna\s1.log 2>&1
echo s1=%ERRORLEVEL%

echo ===== S2: spec 4, min-p 0.00 (never truncate) =====
strata.exe %BASE% --spec 4 --spec-min-p 0.0 > C:\Users\rr\Desktop\Edge0-rdna\s2.log 2>&1
echo s2=%ERRORLEVEL%

echo ===== S3: spec 6, min-p 0.50 (wider window, same gate) =====
strata.exe %BASE% --spec 6 --spec-min-p 0.5 > C:\Users\rr\Desktop\Edge0-rdna\s3.log 2>&1
echo s3=%ERRORLEVEL%

echo ===== S4: spec 6, min-p 0.20 =====
strata.exe %BASE% --spec 6 --spec-min-p 0.2 > C:\Users\rr\Desktop\Edge0-rdna\s4.log 2>&1
echo s4=%ERRORLEVEL%

echo SPEC_DONE
endlocal
