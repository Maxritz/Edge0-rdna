@echo off
rem Fewer misses => less CPU pool.  Does releasing the VRAM reserve buy cache slots and decode speed?
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --gpu 0 --max-new 64
strata.exe %BASE% --pcie-frac 0.4 --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\rv512a.log 2>&1
strata.exe %BASE% --pcie-frac 0.4 --vram-reserve-mib 256 > C:\Users\rr\Desktop\Edge0-rdna\rv256.log 2>&1
strata.exe %BASE% --pcie-frac 0.4 --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\rv512b.log 2>&1
strata.exe %BASE% --pcie-frac 0.4 --vram-reserve-mib 0 > C:\Users\rr\Desktop\Edge0-rdna\rv0.log 2>&1
strata.exe %BASE% --pcie-frac 0 --vram-reserve-mib 0 > C:\Users\rr\Desktop\Edge0-rdna\rv0f0.log 2>&1
strata.exe %BASE% --pcie-frac 0 --vram-reserve-mib 512 > C:\Users\rr\Desktop\Edge0-rdna\rv512f0.log 2>&1
echo RV_DONE
endlocal