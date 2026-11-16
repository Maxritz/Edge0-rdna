@echo off
rem The decode D stage is 21.46 ms a round and reads 52 MB a layer (2.5 GB a round) at ~150 GB/s against
rem DRAM's ~384.  latent_d_fast keeps D's own shape: one warp per output row (coalesced 128-byte row
rem reads), T tokens accumulated in registers from each loaded weight word (4x reuse), no staging, no row
rem padding - and it is bit-exact to the per-row reference (latent_u_tile_test: mean_rel 0.000).
rem   D0 iq_mmvq   D2 latent_d_fast   D0   D2
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

echo ===== D0: iq_mmvq =====
set STRATA_VERIFY_D_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q0.log 2>&1
echo q0=%ERRORLEVEL%
echo ===== D2: latent_d_fast =====
set STRATA_VERIFY_D_TILE=2
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q2.log 2>&1
echo q2=%ERRORLEVEL%
echo ===== D0: iq_mmvq =====
set STRATA_VERIFY_D_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q0b.log 2>&1
echo q0b=%ERRORLEVEL%
echo ===== D2: latent_d_fast =====
set STRATA_VERIFY_D_TILE=2
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q2b.log 2>&1
echo q2b=%ERRORLEVEL%
echo Q_DONE
endlocal