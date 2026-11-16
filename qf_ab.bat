@echo off
rem hc1 up is 24.61 ms/round in the engine but 2.7 ms/round in the isolated bench (48 cold layer-sets, T=4,
rem 56.8 us a layer).  down (2.89 vs 3.33 predicted) and norm (0.30 vs 0.54) MATCH the bench, so only the up
rem stage is anomalous.  The bench exercises fused_gr_read_multi; the engine's log says it runs the STAGED read,
rem whose up branch is a different kernel (gr_up_q32_kernel).  These arms change only that branch:
rem   A default                       -> gr_up_q32_kernel (32 columns, 512 threads, quantizes q8_1 in block)
rem   B STRATA_QFUSE_HC_LOCAL=0       -> gr_up_multi_kernel + gr_q8_tail (the counter/fence tail)
rem   C STRATA_QFUSE_HC=0             -> no q8_1 image at all, gr_up_multi_kernel
rem Metric: the hc1 up / hc1 down / hc1 norm slots, decode tok/s, and the 32-token output as the gate.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== A default =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\qf1.log 2>&1
echo a=%ERRORLEVEL%
echo ===== B QFUSE_HC_LOCAL=0 =====
set STRATA_QFUSE_HC_LOCAL=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\qf2.log 2>&1
echo b=%ERRORLEVEL%
echo ===== C QFUSE_HC=0 =====
set STRATA_QFUSE_HC_LOCAL=
set STRATA_QFUSE_HC=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\qf3.log 2>&1
echo c=%ERRORLEVEL%
echo ===== A again =====
set STRATA_QFUSE_HC=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\qf4.log 2>&1
echo d=%ERRORLEVEL%
echo QF_DONE
endlocal