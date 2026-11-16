@echo off
rem Three HC read variants, all checked bit-for-bit against the plain read on this card, none of which the
rem isolated bench ever covered (fused_gr_bench calls fused_gr_read_multi only).  The staged variant is the
rem shipped default and charges hc1 up = 18.4 ms a round; hc1 down on the same 6.2 MB is 2.16 and the isolated
rem kernel is 0.057 ms a layer.  Persist OFF so the up slot is stamped.  Gate: identical 32-token output.
rem   S0 STRATA_HC_SPLIT=0  the plain read (MAIN's original: one kernel per stage, single-token shaped)
rem   S1 STRATA_HC_SPLIT=1  the split read
rem   S2 default (2)        the staged read
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
set STRATA_HC_PERSIST=0
set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill 512 --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --pcie-frac 0.1 --gpu 0 --max-new 32 --stats --vram-reserve-mib 512
echo ===== S0 plain =====
set STRATA_HC_SPLIT=0
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\sv0.log 2>&1
echo a=%ERRORLEVEL%
echo ===== S1 split =====
set STRATA_HC_SPLIT=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\sv1.log 2>&1
echo b=%ERRORLEVEL%
echo ===== S2 staged (default) =====
set STRATA_HC_SPLIT=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\sv2.log 2>&1
echo c=%ERRORLEVEL%
echo SV_DONE
endlocal