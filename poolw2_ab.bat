@echo off
rem Where does the CPU pool saturate?  Measured so far (pool's own reported bandwidth over its rows
rem phases, 32-token runs): 3 workers/4 threads 29.8 GB/s, 5 workers/6 threads 30.5, 6 workers/7 threads
rem 29.0 - a plateau, i.e. the machine's DDR4 read ceiling rather than a kernel limit.  4 workers gave
rem 20.2, which is either a placement artifact or noise; this run repeats it between two 5-worker arms.
rem   A5 / A4 / A5 / A4 / A2
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

echo ===== A5: 5 workers =====
strata.exe %BASE% --pool-workers 5 > C:\Users\rr\Desktop\Edge0-rdna\q5a.log 2>&1
echo q5a=%ERRORLEVEL%
echo ===== A4: 4 workers =====
strata.exe %BASE% --pool-workers 4 > C:\Users\rr\Desktop\Edge0-rdna\q4a.log 2>&1
echo q4a=%ERRORLEVEL%
echo ===== A5: 5 workers =====
strata.exe %BASE% --pool-workers 5 > C:\Users\rr\Desktop\Edge0-rdna\q5b.log 2>&1
echo q5b=%ERRORLEVEL%
echo ===== A4: 4 workers =====
strata.exe %BASE% --pool-workers 4 > C:\Users\rr\Desktop\Edge0-rdna\q4b.log 2>&1
echo q4b=%ERRORLEVEL%
echo ===== A2: 2 workers =====
strata.exe %BASE% --pool-workers 2 > C:\Users\rr\Desktop\Edge0-rdna\q2a.log 2>&1
echo q2a=%ERRORLEVEL%
echo Q_DONE
endlocal
