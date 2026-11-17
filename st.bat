@echo off
rem Does gemm gate/up track the EXPERT STREAM?  chunk 512 vs 2048 halves the streamed blobs.
rem If gate/up is flat while blobs halve, the A stage is compute/stage-bound and no byte work can help it.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 8192 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
strata.exe %BASE% --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\st512a.log 2>&1
strata.exe %BASE% --prefill 2048 > C:\Users\rr\Desktop\Edge0-rdna\st2048a.log 2>&1
strata.exe %BASE% --prefill 512 > C:\Users\rr\Desktop\Edge0-rdna\st512b.log 2>&1
strata.exe %BASE% --prefill 2048 > C:\Users\rr\Desktop\Edge0-rdna\st2048b.log 2>&1
echo ST_DONE
endlocal