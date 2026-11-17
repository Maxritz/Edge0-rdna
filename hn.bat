@echo off
rem Q4 with the GPU hit path disabled, two ways: --no-token-graph and --no-pool.
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
set STRATA_PREFILL_TIMING=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== Q4 --no-token-graph (no MTP, to isolate) =====
strata.exe %BASE% --no-token-graph > C:\Users\rr\Desktop\Edge0-rdna\hn_ntg.log 2>&1
echo a=%ERRORLEVEL%
echo ===== Q4 --no-pool =====
strata.exe %BASE% --no-pool > C:\Users\rr\Desktop\Edge0-rdna\hn_np.log 2>&1
echo b=%ERRORLEVEL%
echo ===== Q4 control =====
strata.exe --mtp E:\Edge0-120B-q3k\mtp %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hn_ctl.log 2>&1
echo c=%ERRORLEVEL%
echo HN_DONE
endlocal