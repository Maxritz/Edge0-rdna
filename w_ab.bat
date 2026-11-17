@echo off
rem gr_up_wide_kernel on gfx1031, which could not be measured before because gr_up_wide_on() returned false
rem under __HIPCC__ BEFORE reading STRATA_GR_UP_WIDE.  The wide kernel is only selectable when q8_mixed is null,
rem so the arm needs STRATA_QFUSE_HC=0 (measured neutral on its own), and persist must be off so the up slot is
rem stamped rather than folded into hc1 norm.  One warp a row, 1280 blocks instead of 160: 8x the independent
rem warps, which is what a small-M read is short of.
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
set STRATA_HC_PERSIST=0
set STRATA_QFUSE_HC=0
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== W0 narrow (multi) =====
set STRATA_GR_UP_WIDE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w0.log 2>&1
echo a=%ERRORLEVEL%
echo ===== W1 wide =====
set STRATA_GR_UP_WIDE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w1.log 2>&1
echo b=%ERRORLEVEL%
echo ===== W0 again =====
set STRATA_GR_UP_WIDE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w0b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== W1 again =====
set STRATA_GR_UP_WIDE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w1b.log 2>&1
echo d=%ERRORLEVEL%
echo W_DONE
endlocal