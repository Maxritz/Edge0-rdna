@echo off
rem The hc1 up split is only visible with the persistent read OFF (it charges the whole read to `hc1 norm`).
rem Per layer, at T=4: the engine charges up 522 us, down 61 us, norm 6.3 us; the isolated bench gives
rem up 56.8, down 68.6, norm 11.2.  down and norm MATCH, up is 9.2x.  Both read 6.55 MB.  So the up BRANCH is
rem the anomaly.  These arms keep persist off and change only the up branch:
rem   A default                    -> gr_up_q32_kernel (512 threads, 32 cols, q8_1 in block)
rem   B STRATA_QFUSE_HC_LOCAL=0    -> gr_up_multi_kernel + gr_q8_tail (256 threads, 16 cols, counter tail)
rem   C STRATA_HC_SPLIT=1          -> the older split read
rem Gate: 32-token output identical; metric: hc1 up, and decode tok/s.
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
echo ===== U1 default up branch =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\up1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== U2 QFUSE_HC_LOCAL=0 (multi + q8_tail) =====
set STRATA_QFUSE_HC_LOCAL=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\up2.log 2>&1
echo b=%ERRORLEVEL%
echo ===== U3 default again =====
set STRATA_QFUSE_HC_LOCAL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\up3.log 2>&1
echo c=%ERRORLEVEL%
echo UP_DONE
endlocal