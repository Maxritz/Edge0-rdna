@echo off
rem The 14.4 ms "matvec+hits" bucket is nearly all latent_u_matvec (the copy is 0.19-0.25 ms).  It now has
rem its own slot (35, with the copy on 29 and the hits left on 23), and this pass A/Bs it against
rem latent_u_grouped - the kernel the GPU rows already use (126 GB/s against the matvec's 81), which the
rem moe_hit_set assignment makes safe: the GPU rows are overwritten from hit_out right after.
rem   M0 matvec (shipped)   M1 grouped   M0   M1
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
set STRATA_VERIFY_U_PROF=1

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats

echo ===== M0: matvec =====
set STRATA_VERIFY_U_CPU_GROUPED=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\m0.log 2>&1
echo m0=%ERRORLEVEL%
echo ===== M1: grouped =====
set STRATA_VERIFY_U_CPU_GROUPED=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\m1.log 2>&1
echo m1=%ERRORLEVEL%
echo ===== M0: matvec =====
set STRATA_VERIFY_U_CPU_GROUPED=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\m0b.log 2>&1
echo m0b=%ERRORLEVEL%
echo ===== M1: grouped =====
set STRATA_VERIFY_U_CPU_GROUPED=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\m1b.log 2>&1
echo m1b=%ERRORLEVEL%
echo M_DONE
endlocal