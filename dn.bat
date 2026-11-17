@echo off
rem What does ONE dependent kernel node cost in a decode window?  STRATA_DUMMY_NODES=k adds k empty nodes/layer.
rem If 4 nodes cost ~4 x 0.23 ms, the hc1 slot's fixed cost is node latency (fuse kernels).  If ~0, it is a wait.
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
set STRATA_VERIFY_PROFILE=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 64 --vram-reserve-mib 512
set STRATA_DUMMY_NODES=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dn0.log 2>&1
echo a=%ERRORLEVEL%
set STRATA_DUMMY_NODES=4
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dn4.log 2>&1
echo b=%ERRORLEVEL%
set STRATA_DUMMY_NODES=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dn8.log 2>&1
echo c=%ERRORLEVEL%
echo DN_DONE
endlocal