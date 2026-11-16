@echo off
rem SHERLOCK step 1: does the GPU actually work during a decode round, or does it idle waiting for the host?
rem STRATA_VERIFY_TRACE=1 prints (a) the host handshake ring: WINDOW / LAUNCHED / RANG k l <us> / SERVED k l
rem and (b) the GPU breadcrumbs: us after the window's first stamp at which the GPU passed each layer's stamp.
rem If RANG is microseconds and the breadcrumbs advance smoothly -> GPU-bound.  If RANG is ~7 ms per layer
rem -> the GPU finished its layer and sat idle until the host served the next one: host-bound handshake.
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
set STRATA_VERIFY_TRACE=1
set STRATA_ROUTE_STATS=1

strata.exe --pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 16 --stats > C:\Users\rr\Desktop\Edge0-rdna\tr_dec.log 2>&1
echo exit=%ERRORLEVEL%
endlocal
