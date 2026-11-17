@echo off
rem Is the intermittent acceptance collapse a side-stream race?  Same config, side stream OFF, 6 runs.
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
set STRATA_MTP_SIDE_STREAM=0
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 6 --spec-min-p 0.3 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --gpu 0 --max-new 64 --vram-reserve-mib 512 --pcie-frac 0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ns1.log 2>&1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ns2.log 2>&1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ns3.log 2>&1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ns4.log 2>&1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ns5.log 2>&1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ns6.log 2>&1
echo NS_DONE
endlocal