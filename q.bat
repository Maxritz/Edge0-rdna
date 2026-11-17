@echo off
rem Why is the FFN-side hc read 32 GB/s while the attention-side read is 239 GB/s for the SAME bytes?
rem Four shipped variants, env only: the consumer-write (QFUSE), the local-epilogue, the wide up, the persistent read.
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build
set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q4.bin
set STRATA_LATENT_DENSE_FMT=
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_MTP_SIDE_STREAM=1
set STRATA_VERIFY_PROFILE=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --gpu 0 --max-new 64 --vram-reserve-mib 512
set STRATA_QFUSE=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q_def.log 2>&1
set STRATA_QFUSE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q_off.log 2>&1
set STRATA_QFUSE=
set STRATA_QFUSE_HC_LOCAL=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q_loc.log 2>&1
set STRATA_QFUSE_HC_LOCAL=
set STRATA_GR_UP_WIDE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q_wide.log 2>&1
echo Q_DONE
endlocal