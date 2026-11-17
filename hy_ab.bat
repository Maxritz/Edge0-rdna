@echo off
rem The auto-chunk cliff, A/B'd in alternating pairs in ONE session (same binary, env only), so a VRAM drift
rem cannot be mistaken for the change.  The scan demanded a FULL staging ring; STRATA_PREFILL_RING_HYST=0 keeps
rem that rule, unset/1 accepts three quarters.  At 720-742 cache slots the full-ring rule drops the automatic
rem chunk to 256, and chunk 256 measures 167 tok/s against 236-239 at 512.
rem Gate: the 32-token output must be identical between arms (chunk size cannot change the arithmetic).
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build
set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_MTP_SIDE_STREAM=1
set STRATA_PREFILL_TIMING=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== H0 full-ring rule (old) =====
set STRATA_PREFILL_RING_HYST=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hy0.log 2>&1
echo a=%ERRORLEVEL%
echo ===== H1 three-quarters (new) =====
set STRATA_PREFILL_RING_HYST=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hy1.log 2>&1
echo b=%ERRORLEVEL%
echo ===== H0 again =====
set STRATA_PREFILL_RING_HYST=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hy0b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== H1 again =====
set STRATA_PREFILL_RING_HYST=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\hy1b.log 2>&1
echo d=%ERRORLEVEL%
echo HY_DONE
endlocal