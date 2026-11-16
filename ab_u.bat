@echo off
rem E/F: the U stage's per-row kernel (STRATA_LATENT_U_OLD=1) vs the weight-reuse kernel (default).
rem Same flags on both arms.  The reuse kernel keeps a column's weights in registers across every routed
rem row of the group, with the per-output arithmetic unchanged, so the tokens must be identical.
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

echo ===== ARM E: U per-row (old) =====
set STRATA_LATENT_U_OLD=1
strata.exe %COMMON% > C:\Users\rr\Desktop\Edge0-rdna\ab_uold.log 2>&1
echo exitE=%ERRORLEVEL%

echo ===== ARM F: U weight-reuse (new) =====
set STRATA_LATENT_U_OLD=0
strata.exe %COMMON% > C:\Users\rr\Desktop\Edge0-rdna\ab_unew.log 2>&1
echo exitF=%ERRORLEVEL%

echo EF_DONE
endlocal
