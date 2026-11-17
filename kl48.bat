@echo off
rem KL gate with the fmt-aware tile kernels: Q8 vs Q4 table, same prompt, 8192 ctx, 128 new.
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build
set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_MTP_SIDE_STREAM=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 8192 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 128 --stats --vram-reserve-mib 512
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_LATENT_DENSE_FMT=8
set STRATA_LOGITS_DUMP=C:\Users\rr\Desktop\Edge0-rdna\kg8.bin
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\kg8.log 2>&1
echo q8=%ERRORLEVEL%
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q4.bin
set STRATA_LATENT_DENSE_FMT=4
set STRATA_LOGITS_DUMP=C:\Users\rr\Desktop\Edge0-rdna\kg4.bin
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\kg4.log 2>&1
echo q4=%ERRORLEVEL%
echo KL_DONE
endlocal