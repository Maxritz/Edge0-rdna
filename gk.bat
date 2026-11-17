@echo off
rem Does the decode expert lane change the 512K+MTP token output?  Gate config, default vs --pcie-frac 0.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 2 --mtp E:\Edge0-120B-q3k\mtp --max-context 524288 --kv int8 --kv-resident 1 --gpu 0 --max-new 4 --vram-reserve-mib 512
strata.exe %BASE% --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt.txt > C:\Users\rr\Desktop\Edge0-rdna\gk_def.log 2>&1
strata.exe %BASE% --pcie-frac 0 --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt.txt > C:\Users\rr\Desktop\Edge0-rdna\gk_f0.log 2>&1
echo GK_DONE
endlocal