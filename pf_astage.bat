@echo off
rem The prefill A stage, three implementations, one variable each. %1 = tag, %2/%3 = the two knobs to set.
rem   fused : STRATA_LAT_FUSED_Q3K=1   (lat_q3k_fused, what bench4k has been running)
rem   mmq   : STRATA_LAT_FUSED_Q3K=0   (native_expert_grouped / MMQ's own A stage)
rem   mmqm  : + STRATA_LATENT_MMQ=1    (the MMQ A stage with its gather)
setlocal
set "ROOT=C:\Users\rr\Desktop\Edge0-rdna"
cd /d "%ROOT%\edge0-strata\build"

set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_MTP_SIDE_STREAM=1
set %2
set %3
set STRATA_PREFILL_TIMING=1

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
  --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt ^
  --max-context 4096 ^
  --kv int8 ^
  --vram-reserve-mib 512 ^
  --gpu 0 ^
  --max-new 16 ^
  --stats ^
  > "%ROOT%\pf_astage_%1.log" 2>&1

endlocal
