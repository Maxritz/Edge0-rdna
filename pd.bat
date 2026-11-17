@echo off
rem The engine default for a native pack is pcie-frac 0.55; my harness has hard-pinned 0.10.  Who is right?
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --gpu 0 --max-new 64 --vram-reserve-mib 512
set STRATA_PCIE_FRAC=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pd_def_a.log 2>&1
set STRATA_PCIE_FRAC=0.40
strata.exe %BASE% --pcie-frac 0.40 > C:\Users\rr\Desktop\Edge0-rdna\pd40a.log 2>&1
set STRATA_PCIE_FRAC=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pd_def_b.log 2>&1
set STRATA_PCIE_FRAC=0.40
strata.exe %BASE% --pcie-frac 0.40 > C:\Users\rr\Desktop\Edge0-rdna\pd40b.log 2>&1
echo PD_DONE
endlocal