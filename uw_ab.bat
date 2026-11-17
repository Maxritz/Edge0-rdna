@echo off
rem STRATA_LATENT_U_WARPS: output columns a block covers, one warp each.  latent.cu's note says decode's U is
rem bound by BLOCK COUNT (320 blocks a routed row x 10 rows x 48 layers ~ 154k blocks a round at ~86 ns), that
rem this is the knob that matters, and that every width is bit-identical (same terms, same order, same reduction
rem tree).  Default 8.  Gate: 32-token output identical.  Slots 33/34/35 need STRATA_VERIFY_U_PROF=1.
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
set STRATA_VERIFY_U_PROF=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== W8 (default) =====
set STRATA_LATENT_U_WARPS=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\uw8.log 2>&1
echo a=%ERRORLEVEL%
echo ===== W16 =====
set STRATA_LATENT_U_WARPS=16
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\uw16.log 2>&1
echo b=%ERRORLEVEL%
echo ===== W32 =====
set STRATA_LATENT_U_WARPS=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\uw32.log 2>&1
echo c=%ERRORLEVEL%
echo ===== W8 again =====
set STRATA_LATENT_U_WARPS=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\uw8b.log 2>&1
echo d=%ERRORLEVEL%
echo UW_DONE
endlocal