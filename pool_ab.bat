@echo off
rem Decode's remaining 24.7 ms/window is the CPU pool: 731 MB of expert weights per round from DDR4.
rem The pool reports 30.5 GB/s over its rows phases, against ~40 GB/s of realistic dual-channel DDR4-3200
rem read - so there is room, if the workers are not the limiter.  Two knobs, both existing:
rem   --pool-tasks N   batched CPU expert tasks per phase (0 = the policy's 18)
rem   --pool-workers N workers (0 = every physical core but the host's = 5 on this 5600X)
rem Watch the "pool multi" line (GB/s and ms/round) and decode tok/s.
rem   P0 baseline   P1 tasks 36   P2 9 workers + tasks 36
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== P0: baseline (5 workers, 18 tasks) =====
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\w0.log 2>&1
echo w0=%ERRORLEVEL%

echo ===== P1: tasks 36 =====
strata.exe %BASE% --pool-tasks 36 > C:\Users\rr\Desktop\Edge0-rdna\w1.log 2>&1
echo w1=%ERRORLEVEL%

echo ===== P2: 9 workers + tasks 36 =====
strata.exe %BASE% --pool-workers 9 --pool-tasks 36 > C:\Users\rr\Desktop\Edge0-rdna\w2.log 2>&1
echo w2=%ERRORLEVEL%

echo W_DONE
endlocal
