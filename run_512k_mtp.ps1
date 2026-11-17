# Run one 512K + MTP configuration and capture everything, no .bat in between:
#   powershell -File run_512k_mtp.ps1 -Embed 1 -Tag a_embed1
param(
    [string]$Embed = '1',
    [string]$Tag = 'run',
    [string]$Kv = 'turbo3',
    [int]$KvResident = 65536,
    [int]$Reserve = 300,
    [int]$Mtp = 1,
    [int]$MaxContext = 524288,
    [string]$Prefill = 'auto',
    [string]$Tokens = 'prompt64.txt',
    [string]$Dense = 'E:/Edge0-120B-q3k/latent_dense_q8.bin',
    [string]$DenseFmt = '8',
    [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\rr\Desktop\Edge0-rdna'
$build = Join-Path $root 'edge0-strata\build'

$env:STRATA_LATENT_A = 'E:/Edge0-120B-q3k/experts.bin'
$env:STRATA_LATENT_LABELS = 'E:/Edge0-120B-q3k/labels.bin'
$env:STRATA_LATENT_DENSE = $Dense
$env:STRATA_LATENT_DENSE_FMT = $DenseFmt
$env:STRATA_EMBED_VRAM = $Embed
$env:STRATA_LATENT_FUSED = '1'
$env:STRATA_LATENT_FUSED_DU = '1'
$env:STRATA_LATENT_U_GPU = '1'
$env:STRATA_LAT_FUSED_Q3K = '1'
$env:STRATA_MTP_SIDE_STREAM = '1'
$env:STRATA_TRACE = '1'

$log = Join-Path $root "ctx_mtp_$Tag.log"
$args = @(
    '--pack', 'E:\Edge0-120B-q3k',
    '--native', 'E:\Edge0-120B-q3k\model-skeleton.gguf',
    '--native-dense-gguf', 'E:\Edge0-120B-q3k\model-skeleton.gguf',
    '--ple-gguf', 'E:\Edge0-120B-q3k\ple-fp8.gguf',
    '--ple-io', 'direct',
    '--expert-profile', 'E:\Edge0-120B-q3k\profile.bin',
    '--expert-cache', 'auto',
    '--prefill', "$Prefill",
    '--spec', '2',
    '--tokens-file', (Join-Path $root $Tokens),
    '--max-context', "$MaxContext",
    '--kv', $Kv,
    '--kv-resident', "$KvResident",
    '--vram-reserve-mib', "$Reserve",
    '--gpu', '0',
    '--max-new', '4'
)
if ($Extra -ne '') { $args += $Extra.Split(' ') }
if ($Mtp -ne 0) { $args += @('--mtp', 'E:\Edge0-120B-q3k\mtp') }

$t0 = Get-Date
$p = Start-Process -FilePath (Join-Path $build 'strata.exe') -ArgumentList $args -WorkingDirectory $build `
    -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
$sec = [int]((Get-Date) - $t0).TotalSeconds
Get-Content "$log.err" | Set-Content $log
Get-Content "$log.out" | Add-Content $log
Remove-Item "$log.out", "$log.err"
Write-Output "=== $Tag exit $($p.ExitCode) after ${sec}s -> $log ==="
