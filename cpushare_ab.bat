@echo off
rem The A stage (5,619 ms of the 20,397 ms prompt) is streamed-expert-bound, not MAC-bound: at chunk 1024
rem the tile is twice as full (groups ~50 instead of ~25) and gemm gate/up got 70 %% WORSE while the
rem blobs rose 28 %% (5,982 -> 10,196 ms; blobs 63,361 -> 80,897).  So the lever is bytes over PCIe, and
rem the engine already has one: STRATA_PREFILL_CPU_SHARE computes part of the expert set on the CPU pool
rem from the host-resident arena, which never crosses PCIe.  It is off here because the Windows commit
rem limit is tight ("only 1.3 GiB left; the share needs 0.30 GiB and 4.0 GiB left over").
rem   S0 baseline   S1 CPU share auto   S2 CPU share auto (second pair)
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 8 --stats

echo ===== S0: baseline =====
set STRATA_PREFILL_CPU_SHARE=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\s0.log 2>&1
echo s0=%ERRORLEVEL%

echo ===== S1: CPU share auto =====
set STRATA_PREFILL_CPU_SHARE=auto
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\s1.log 2>&1
echo s1=%ERRORLEVEL%

echo ===== S2: CPU share auto (second pair) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\s2.log 2>&1
echo s2=%ERRORLEVEL%
set STRATA_PREFILL_CPU_SHARE=
echo S_DONE
endlocal
