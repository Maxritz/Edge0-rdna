@echo off
rem Context ceiling scan. %1 = kv format, %2 = max-context, %3 = tag for the log.
rem A 64-token prompt keeps prefill out of it: --max-context sizes the K/V and the state, so the VRAM
rem verdict is the same whether the prompt is 64 or 32000 tokens - and it costs 30 s instead of 7 min.
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
set STRATA_LAT_FUSED_Q3K=1
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
  --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt ^
  --max-context %2 ^
  --kv %1 ^
  --vram-reserve-mib 512 ^
  --gpu 0 ^
  --max-new 4 ^
  %4 %5 %6 %7 %8 %9 > "%ROOT%\ctx_%3.log" 2>&1

endlocal
