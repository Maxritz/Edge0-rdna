@echo off
rem The decode-shaped U tile (STRATA_VERIFY_U_TILE=2) against the shipped per-row kernel (0), alternating.
rem cap is 10 rows a layer, so this issues ~150 blocks a layer instead of ~3,200 and stages a label's
rem slice once for up to 16 rows.  The [T4] table's "u: gemv" is the metric (13.23 ms with the per-row
rem kernel); the 32-token `output` is the identity gate.
rem   U0 per-row   U2 small tile   U0   U2
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
set STRATA_VERIFY_U_PROF=1
set STRATA_VERIFY_PROFILE=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== U0: per-row =====
set STRATA_VERIFY_U_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w0u.log 2>&1
echo u0=%ERRORLEVEL%
echo ===== U2: decode-shaped tile =====
set STRATA_VERIFY_U_TILE=2
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w2u.log 2>&1
echo u2=%ERRORLEVEL%
echo ===== U0: per-row =====
set STRATA_VERIFY_U_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w0v.log 2>&1
echo u0b=%ERRORLEVEL%
echo ===== U2: decode-shaped tile =====
set STRATA_VERIFY_U_TILE=2
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w2v.log 2>&1
echo u2b=%ERRORLEVEL%
echo WU_DONE
endlocal
