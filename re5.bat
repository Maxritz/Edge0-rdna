@echo off
rem A: the shipped path (no profiler, which disables three fusions) - the decode number that counts.
rem B: the same with STRATA_VERIFY_PROFILE, to read the new uWait / uCopy+U split of the 43%% bucket.
rem STRATA_HC_PERSIST is NOT set: on gfx10 it must now default on by itself (fused_gr.cu arch_default_on).
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== A: shipped path, HC_PERSIST default =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\re5a.log 2>&1
echo a=%ERRORLEVEL%

echo ===== B: profiled, for the uWait / uCopy split =====
set STRATA_VERIFY_PROFILE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\re5b.log 2>&1
echo b=%ERRORLEVEL%
set STRATA_VERIFY_PROFILE=

echo RE5_DONE
endlocal
