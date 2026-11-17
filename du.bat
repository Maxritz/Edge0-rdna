@echo off
rem D/U attribution: split gemm down (q8_1(H)+U) from gather (D) and the two hc reads.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --prefill 512 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats --vram-reserve-mib 512
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\du_base1.log 2>&1
set STRATA_LATENT_FUSED_DU=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\du_nf1.log 2>&1
set STRATA_DU_PROF=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\du_prof.log 2>&1
set STRATA_LATENT_U_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\du_notile.log 2>&1
set STRATA_LATENT_U_TILE=
set STRATA_DU_PROF=
set STRATA_LATENT_FUSED_DU=
set STRATA_LATENT_U_GPU=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\du_cpu.log 2>&1
set STRATA_LATENT_U_GPU=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\du_base2.log 2>&1
echo DU_DONE
endlocal