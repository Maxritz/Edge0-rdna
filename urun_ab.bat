@echo off
rem The U projection is 9.1 s of the 28.5 s prompt (32 %): 23.7 ms a layer, and it re-reads each group's
rem whole U matrix once per routed row (~20 GB a layer through the Infinity Cache at ~850 GB/s,
rem 1.42 TFLOP/s = 2 % of the card's int8 peak).  STRATA_LATENT_U_RUN=R runs the weight-reuse kernel:
rem one block per RUN of R routed rows that share a label, an output column's weights held in registers.
rem The arithmetic per output is the per-row kernel's, in the same term order, so the outputs must be
rem BIT-IDENTICAL - the 8 tokens here are the gate, and [uprof] gives the per-layer ms to compare with
rem 23.7.
rem   U0 R=0 (per-row, the shipped kernel)   U1 R=4   U2 R=8   U3 R=16   U4 R=32
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
set STRATA_DU_PROF=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats

echo ===== U0: per-row (shipped) =====
set STRATA_LATENT_U_RUN=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\u0.log 2>&1
echo u0=%ERRORLEVEL%

echo ===== U1: R=4 =====
set STRATA_LATENT_U_RUN=4
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\u1.log 2>&1
echo u1=%ERRORLEVEL%

echo ===== U2: R=8 =====
set STRATA_LATENT_U_RUN=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\u2.log 2>&1
echo u2=%ERRORLEVEL%

echo ===== U3: R=16 =====
set STRATA_LATENT_U_RUN=16
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\u3.log 2>&1
echo u3=%ERRORLEVEL%

echo ===== U4: R=32 =====
set STRATA_LATENT_U_RUN=32
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\u4.log 2>&1
echo u4=%ERRORLEVEL%
set STRATA_LATENT_U_RUN=
echo U_DONE
endlocal
