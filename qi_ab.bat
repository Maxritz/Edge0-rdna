@echo off
rem Residency is now RULED OUT (both w_up and w_down report DEVICE (VRAM), 6.2 MB each).  The remaining candidate
rem for hc1 up = 18.3 ms against hc1 down = 2.16 ms on the same bytes, same kernel, token-independent: the q8_1
rem image the up stage writes (half=1 to nat_xq_, half=0 to xq_), plus its quantize.  STRATA_QFUSE_HC=0 makes
rem q8_mixed null for both halves, so the up kernel runs with no image.  Persist must be OFF or up is folded into
rem `norm`.  Gate: 32-token output identical.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== Q1 QFUSE on (default) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\qi1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== Q2 QFUSE_HC=0 (no q8_1 image) =====
set STRATA_QFUSE_HC=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\qi2.log 2>&1
echo b=%ERRORLEVEL%
echo ===== Q3 QFUSE on again =====
set STRATA_QFUSE_HC=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\qi3.log 2>&1
echo c=%ERRORLEVEL%
echo QI_DONE
endlocal