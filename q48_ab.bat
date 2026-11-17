@echo off
rem Q4_0 latent D/U tables vs Q8_0: the decode lever.  Alternating pairs, chunk pinned 512, output checked.
rem KL gate already run: mean KL 9.47e-3 (7.6x the 1.25e-3 run-to-run floor), top-1 100 % over the coherent prefix.
setlocal
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build
set STRATA_LATENT_A=E:/Edge0-120B-q3k/experts.bin
set STRATA_LATENT_LABELS=E:/Edge0-120B-q3k/labels.bin
set STRATA_EMBED_VRAM=1
set STRATA_LATENT_FUSED=1
set STRATA_LATENT_FUSED_DU=1
set STRATA_LATENT_U_GPU=1
set STRATA_LAT_FUSED_Q3K=1
set STRATA_MTP_SIDE_STREAM=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== Q4 =====
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q4.bin
set STRATA_LATENT_DENSE_FMT=4
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q4a.log 2>&1
echo a=%ERRORLEVEL%
echo ===== Q8 =====
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_LATENT_DENSE_FMT=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q8a.log 2>&1
echo b=%ERRORLEVEL%
echo ===== Q4 again =====
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q4.bin
set STRATA_LATENT_DENSE_FMT=4
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q4b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== Q8 again =====
set STRATA_LATENT_DENSE=E:/Edge0-120B-q3k/latent_dense_q8.bin
set STRATA_LATENT_DENSE_FMT=8
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\q8b.log 2>&1
echo d=%ERRORLEVEL%
echo Q48_DONE
endlocal