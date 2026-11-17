@echo off
rem Verify the shipped auto path: 8k auto must reach ~341 (was 265), 4k explicit 512 must hold ~302.
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
set STRATA_PREFILL_TIMING=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
strata.exe %BASE% --max-context 8192 --prefill auto > C:\Users\rr\Desktop\Edge0-rdna\v8a.log 2>&1
strata.exe %BASE% --max-context 4096 --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\v4a.log 2>&1
strata.exe %BASE% --max-context 8192 --prefill auto > C:\Users\rr\Desktop\Edge0-rdna\v8b.log 2>&1
strata.exe %BASE% --max-context 4096 --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\v4b.log 2>&1
echo V_DONE
endlocal