@echo off
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build-hip-rdna2

set STRATA_LATENT_A=E:\Edge0-120B-q3k\experts.bin
set STRATA_LATENT_LABELS=E:\Edge0-120B-q3k\labels.bin
set STRATA_LATENT_DENSE=E:\Edge0-120B-q3k\latent_dense_q4.bin
set STRATA_LATENT_DENSE_FMT=4
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_HIP_PROMPT_F16=1
set STRATA_RDNA2_SGEMM=1
set STRATA_MTP_SIDE_STREAM=1

strata.exe --pack E:\Edge0-120B-q3k ^
  --native E:\Edge0-120B-q3k\model-skeleton.gguf ^
  --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf ^
  --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf ^
  --ple-io direct ^
  --expert-profile E:\Edge0-120B-q3k\profile.bin ^
  --expert-cache auto ^
  --prefill auto ^
  --spec 4 ^
  --spec-min-p 0.5 ^
  --mtp E:\Edge0-120B-q3k\mtp ^
  --max-context 131072 ^
  --kv int8 ^
  --vram-reserve-mib 1024 ^
  --pcie-frac 0.1 ^
  --gpu 0 ^
  --short-read 0 ^
  --tokens 1 ^
  --max-new 1 ^
  2>&1