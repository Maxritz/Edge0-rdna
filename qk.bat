@echo off
rem Re-measure the real delta: 1k alternating (default 0 vs explicit 0.4), and 512K with 12 tokens (multi-round).
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
set B1=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --gpu 0 --max-new 64 --vram-reserve-mib 512
strata.exe %B1% > C:\Users\rr\Desktop\Edge0-rdna\q0_a1.log 2>&1
strata.exe %B1% --pcie-frac 0.4 > C:\Users\rr\Desktop\Edge0-rdna\q4_b1.log 2>&1
strata.exe %B1% > C:\Users\rr\Desktop\Edge0-rdna\q0_a2.log 2>&1
strata.exe %B1% --pcie-frac 0.4 > C:\Users\rr\Desktop\Edge0-rdna\q4_b2.log 2>&1
set B2=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 2 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 524288 --kv turbo3 --kv-resident 65536 --gpu 0 --max-new 12 --vram-reserve-mib 300
strata.exe %B2% > C:\Users\rr\Desktop\Edge0-rdna\k0_a1.log 2>&1
strata.exe %B2% --pcie-frac 0.4 > C:\Users\rr\Desktop\Edge0-rdna\k4_b1.log 2>&1
strata.exe %B2% > C:\Users\rr\Desktop\Edge0-rdna\k0_a2.log 2>&1
strata.exe %B2% --pcie-frac 0.4 > C:\Users\rr\Desktop\Edge0-rdna\k4_b2.log 2>&1
echo QK_DONE
endlocal