@echo off
rem DUAL SUBMISSION: at chunk 832 (bytes already minimal), does a deeper pipe/ring overlap the exposed DMA?
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512 --max-context 8192 --prefill 832
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dp_a1.log 2>&1
strata.exe %BASE% --prefill-pipe 2 > C:\Users\rr\Desktop\Edge0-rdna\dp_p2.log 2>&1
strata.exe %BASE% --prefill-pipe 3 > C:\Users\rr\Desktop\Edge0-rdna\dp_p3.log 2>&1
set STRATA_PREFILL_RING=1024
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dp_r1024.log 2>&1
set STRATA_PREFILL_RING=2048
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dp_r2048.log 2>&1
set STRATA_PREFILL_RING=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dp_a2.log 2>&1
echo DP_DONE
endlocal