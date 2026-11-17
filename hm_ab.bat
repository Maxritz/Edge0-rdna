@echo off
rem HIP inherits the per-token hyper-connection read (CUDA batches it).  Batched is opt-in on HIP and
rem the source asserts "bitwise the same sums" - so the output tokens must match exactly.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 64 --vram-reserve-mib 512
echo ===== per-token (current default) =====
set STRATA_HEAD_MIX_MULTI=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hm0a.log 2>&1
echo a=%ERRORLEVEL%
echo ===== batched =====
set STRATA_HEAD_MIX_MULTI=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hm1a.log 2>&1
echo b=%ERRORLEVEL%
echo ===== per-token again =====
set STRATA_HEAD_MIX_MULTI=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hm0b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== batched again =====
set STRATA_HEAD_MIX_MULTI=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hm1b.log 2>&1
echo d=%ERRORLEVEL%
echo HM_DONE
endlocal