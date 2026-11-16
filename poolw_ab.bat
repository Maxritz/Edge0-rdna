@echo off
rem Is the CPU expert pool thread-bound or DDR4-bound?  The pool reports its own bandwidth over the rows
rem phases (30.5 GB/s at 5 workers).  If the aggregate scales with the worker count the kernel is the
rem limit and there is headroom; if it plateaus around 30 the machine is, and no kernel change can help
rem decode's pcie/uWait (which is exactly 731 MB / 30.5 GB/s = 24 ms a round).
rem   W3 / W4 / W5 (default) / W6
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

echo ===== W3 =====
strata.exe %BASE% --pool-workers 3 > C:\Users\rr\Desktop\Edge0-rdna\pw3.log 2>&1
echo w3=%ERRORLEVEL%
echo ===== W4 =====
strata.exe %BASE% --pool-workers 4 > C:\Users\rr\Desktop\Edge0-rdna\pw4.log 2>&1
echo w4=%ERRORLEVEL%
echo ===== W6 =====
strata.exe %BASE% --pool-workers 6 > C:\Users\rr\Desktop\Edge0-rdna\pw6.log 2>&1
echo w6=%ERRORLEVEL%
echo PW_DONE
endlocal
