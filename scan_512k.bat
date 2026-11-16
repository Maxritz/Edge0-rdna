@echo off
rem Can this card reach 512K at all? %1 = kv format, %2 = KV-streaming cells kept in VRAM, %3 = tag.
rem Two levers the engine's own refusal message names, applied together: no MTP draft layer + its 281 MiB head
rem (the largest single consumer that is not needed to read a prompt) and a 300 MiB reserve instead of 512.
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

strata.exe --pack E:\Edge0-120B-q3k ^
  --native E:\Edge0-120B-q3k\model-skeleton.gguf ^
  --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf ^
  --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf ^
  --ple-io direct ^
  --expert-profile E:\Edge0-120B-q3k\profile.bin ^
  --expert-cache auto ^
  --prefill auto ^
  --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt ^
  --max-context 524288 ^
  --kv %1 ^
  --kv-resident %2 ^
  --spec 2 ^
  --vram-reserve-mib 300 ^
  --gpu 0 ^
  --max-new 4 ^
  > "%ROOT%\ctx_512k_%3.log" 2>&1

endlocal
