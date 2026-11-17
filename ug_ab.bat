@echo off
rem The CPU rows' U: verify.cpp:2583-2586 records that latent_u_matvec "walks every entry per block
rem (81 GB/s effective)" while latent_u_grouped - the kernel the GPU rows already use - reaches 126 GB/s, and
rem it already has a knob: STRATA_VERIFY_U_CPU_GROUPED=1.  Both compute U[labels[ids[r]]] . hq[r] for the rows
rem the pool owns, so the 32-token output is the gate.  STRATA_VERIFY_U_PROF=1 opens slots 33/34/35.
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
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== U0 matvec (shipped) =====
set STRATA_VERIFY_U_CPU_GROUPED=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ug0.log 2>&1
echo a=%ERRORLEVEL%
echo ===== U1 grouped =====
set STRATA_VERIFY_U_CPU_GROUPED=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ug1.log 2>&1
echo b=%ERRORLEVEL%
echo ===== U0 again =====
set STRATA_VERIFY_U_CPU_GROUPED=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ug0b.log 2>&1
echo c=%ERRORLEVEL%
echo ===== U1 again =====
set STRATA_VERIFY_U_CPU_GROUPED=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\ug1b.log 2>&1
echo d=%ERRORLEVEL%
echo UG_DONE
endlocal