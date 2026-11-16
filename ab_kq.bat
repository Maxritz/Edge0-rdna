@echo off
rem G/H: the D stage's Q8_0 multi-column kernel.  iq_kernels.cu:965-969 keeps it behind STRATA_KQ_SPLIT=1
rem "until the A/B on the measured cards decides (bitwise the old kernels either way)" - that A/B was never
rem run on gfx1031.  Arm G enables it (U stage already on the reuse kernel); arm H is the control.
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

set COMMON=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== ARM G: KQ_SPLIT=0 (per-column D kernel) =====
set STRATA_KQ_SPLIT=0
strata.exe %COMMON% > C:\Users\rr\Desktop\Edge0-rdna\ab_kqoff.log 2>&1
echo exitG=%ERRORLEVEL%

echo ===== ARM H: KQ_SPLIT=1 (decode-once multi-column D kernel) =====
set STRATA_KQ_SPLIT=1
strata.exe %COMMON% > C:\Users\rr\Desktop\Edge0-rdna\ab_kqon.log 2>&1
echo exitH=%ERRORLEVEL%

echo GH_DONE
endlocal
