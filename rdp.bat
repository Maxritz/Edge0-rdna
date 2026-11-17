@echo off
rem The discriminating experiment both agents converged on: ring depth at a MEANINGFUL scale.
rem 80 slots hold ~3.1 ms of a ~20 ms layer-chunk; 128 was a 60%% change in a 5x-too-small range (hence null).
rem 512 is 6.4x deeper - the largest ring_cap() allows here.  Constant bytes, constant chunk.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
set STRATA_STAGE_SLOTS=80
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rd80a.log 2>&1
echo a=%ERRORLEVEL%
set STRATA_STAGE_SLOTS=512
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rd512a.log 2>&1
echo b=%ERRORLEVEL%
set STRATA_STAGE_SLOTS=80
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rd80b.log 2>&1
echo c=%ERRORLEVEL%
set STRATA_STAGE_SLOTS=512
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\rd512b.log 2>&1
echo d=%ERRORLEVEL%
echo RDP_DONE
endlocal