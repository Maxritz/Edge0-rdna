@echo off
rem The hardware-queue priority question.  On gfx1031 the decode graph streams are hard-disabled
rem (graph_prio_on() returns false for every HIP build) and the MTP draft stream's priority is opt-in
rem (STRATA_MTP_HIP_PRIO, default only on gfx1200/1201).  With STRATA_MTP_SIDE_STREAM=1 the draft stream does
rem run concurrently with the main decode stream, so giving it the hardware queue's top priority is a real,
rem untested lever - the mechanism is AMD's HQD priority through hipStreamCreateWithPriority.
rem Alternating pairs, 32-token output checked on every arm.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== P0 default (no draft-stream priority on this card) =====
set STRATA_MTP_HIP_PRIO=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pr0.log 2>&1
echo a=%ERRORLEVEL%
echo ===== P1 high-priority draft stream =====
set STRATA_MTP_HIP_PRIO=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pr1.log 2>&1
echo b=%ERRORLEVEL%
echo ===== P0 again =====
set STRATA_MTP_HIP_PRIO=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pr0b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== P1 again =====
set STRATA_MTP_HIP_PRIO=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\pr1b.log 2>&1
echo d=%ERRORLEVEL%
echo PR_DONE
endlocal