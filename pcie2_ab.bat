@echo off
rem pcie_frac is the share of cold-miss experts read over PCIe; the rest go to the CPU pool.  The two run
rem in PARALLEL on different bytes, so the round's wait is max(f*B/P, (1-f)*B/30).  Measured: f=0.1 leaves
rem the pool 27.6 ms a round; f=0.55 hands 6.86 experts a layer to the link and the pool drops to 18.8 ms
rem but decode does not improve, i.e. the link is now the long pole.  The balance point with the pool at
rem ~31 GB/s and the link near ~13 GB/s is f = P/(P+30) ~ 0.30, which no harness has tried.
rem   G2 = 0.20   G3 = 0.30   G2 / G3 again (alternating pairs)
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

set BASE=--pack E:\Edge0-120B-q3k --native E:\Edge0-120B-q3k\model-skeleton.gguf --native-dense-gguf E:\Edge0-120B-q3k\model-skeleton.gguf --ple-gguf E:\Edge0-120B-q3k\ple-fp8.gguf --ple-io direct --expert-profile E:\Edge0-120B-q3k\profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --tokens-file C:\Users\rr\Desktop\Edge0-rdna\prompt4k.txt --max-context 4096 --kv int8 --vram-reserve-mib 512 --gpu 0 --max-new 32 --stats

echo ===== G2: 0.20 =====
strata.exe %BASE% --pcie-frac 0.2 > C:\Users\rr\Desktop\Edge0-rdna\g2.log 2>&1
echo g2=%ERRORLEVEL%
echo ===== G3: 0.30 =====
strata.exe %BASE% --pcie-frac 0.3 > C:\Users\rr\Desktop\Edge0-rdna\g3.log 2>&1
echo g3=%ERRORLEVEL%
echo ===== G2: 0.20 =====
strata.exe %BASE% --pcie-frac 0.2 > C:\Users\rr\Desktop\Edge0-rdna\g2b.log 2>&1
echo g2b=%ERRORLEVEL%
echo ===== G3: 0.30 =====
strata.exe %BASE% --pcie-frac 0.3 > C:\Users\rr\Desktop\Edge0-rdna\g3b.log 2>&1
echo g3b=%ERRORLEVEL%
echo G_DONE
endlocal
