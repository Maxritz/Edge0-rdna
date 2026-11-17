@echo off
rem CATCHUP_ALL A/B: 10 runs per arm, interleaved, spec 4 / min-p 0.5.  Scored as a collapse RATE.
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
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_1.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_1.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_2.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_2.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_3.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_3.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_4.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_4.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_5.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_5.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_6.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_6.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_7.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_7.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_8.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_8.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_9.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_9.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_off_10.log 2>&1
set STRATA_MTP_CATCHUP_ALL=1
strata.exe %BASE% > C:\Users\rr\Desktop\Edge0-rdna\cb_on_10.log 2>&1
set STRATA_MTP_CATCHUP_ALL=
echo CB_DONE
endlocal
