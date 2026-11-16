@echo off
rem The decode-side latent D stage runs INSIDE the `hc1 up` slot (verify.cpp 2239-2280, between the FFN read and
rem stamp 32), which is why that slot read 18.4 ms and why nothing about the up kernel moved it: the up is ~2.0 ms
rem (the attention half shows 2.000) and the D stage is ~16.4.  The engine's own note there says the staged tile
rem loses at M=4 "21.3 vs 15.3-17.4 ms", so the three implementations already exist behind a knob.
rem   D0 iq_mmvq        (shipped)
rem   D1 latent_d_tiled (the prefill staged tile)
rem   D2 latent_d_fast  (the decode-shaped kernel, documented bit-exact to the reference)
rem Gate: 32-token output identical.  Metric: hc1 up (= up + D), decode tok/s.
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
set STRATA_VERIFY_PROFILE=1
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== D0 iq_mmvq (shipped) =====
set STRATA_VERIFY_D_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt0.log 2>&1
echo a=%ERRORLEVEL%
echo ===== D2 latent_d_fast =====
set STRATA_VERIFY_D_TILE=2
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt2.log 2>&1
echo b=%ERRORLEVEL%
echo ===== D1 latent_d_tiled =====
set STRATA_VERIFY_D_TILE=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt1.log 2>&1
echo c=%ERRORLEVEL%
echo ===== D0 again =====
set STRATA_VERIFY_D_TILE=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\dt0b.log 2>&1
echo d=%ERRORLEVEL%
echo DT_DONE
endlocal