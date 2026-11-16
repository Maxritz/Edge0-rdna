@echo off
rem The decode D stage is 21.46 ms of the round (slot 17 split: D 21.46, router+plan+shared 1.64).  D reads
rem its whole 52 MB matrix every layer - 2.5 GB a round - at ~120 GB/s against DRAM's ~384, so the metric is
rem bytes moved well.  latent_d_tiled (the kernel that made prefill's D 4.2x faster) moves the same bytes
rem with 2 blocks a CU and 20 chunks of staging; at M = T = 4 its 64-row tile wastes compute but not reads.
rem   T0 iq_mmvq (shipped)   T1 staged tile   T0   T1
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== T0: iq_mmvq =====
set STRATA_VERIFY_D_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\z0.log 2>&1
echo z0=%ERRORLEVEL%
echo ===== T1: staged D tile =====
set STRATA_VERIFY_D_TILE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\z1.log 2>&1
echo z1=%ERRORLEVEL%
echo ===== T0: iq_mmvq =====
set STRATA_VERIFY_D_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\z0b.log 2>&1
echo z0b=%ERRORLEVEL%
echo ===== T1: staged D tile =====
set STRATA_VERIFY_D_TILE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\z1b.log 2>&1
echo z1b=%ERRORLEVEL%
echo Z_DONE
endlocal
