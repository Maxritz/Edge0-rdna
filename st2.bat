@echo off
rem Capture per-round draft probabilities for 6 trials; the collapse is random so we need repeats.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt64.txt --max-context 1024 --kv int8 --gpu 0 --max-new 64 --vram-reserve-mib 512
set STRATA_SPEC_TRACE=C:\Users\rr\Desktop\Edge0-rdna\tr1.txt
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\st1.log 2>&1
set STRATA_SPEC_TRACE=C:\Users\rr\Desktop\Edge0-rdna\tr2.txt
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\st2.log 2>&1
set STRATA_SPEC_TRACE=C:\Users\rr\Desktop\Edge0-rdna\tr3.txt
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\st3.log 2>&1
set STRATA_SPEC_TRACE=C:\Users\rr\Desktop\Edge0-rdna\tr4.txt
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\st4.log 2>&1
set STRATA_SPEC_TRACE=C:\Users\rr\Desktop\Edge0-rdna\tr5.txt
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\st5.log 2>&1
set STRATA_SPEC_TRACE=C:\Users\rr\Desktop\Edge0-rdna\tr6.txt
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\st6.log 2>&1
set STRATA_SPEC_TRACE=
echo ST_DONE
endlocal