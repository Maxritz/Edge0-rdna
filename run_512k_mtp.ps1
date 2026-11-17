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
    [string]$Dense = '',   # '' -> the Q4_0 pack when the converter has been run, else the Q8_0 one
    [string]$DenseFmt = '',
    [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\rr\Desktop\Edge0-rdna'
$build = Join-Path $root 'edge0-strata\build'

$env:STRATA_LATENT_A = 'E:/Edge0-120B-q3k/experts.bin'
$env:STRATA_LATENT_LABELS = 'E:/Edge0-120B-q3k/labels.bin'
# The latent table: the Q4_0 pack is 1.07 GB smaller than Q8_0, which is worth +4.5 % prefill and +4.5 % decode
# directly and, at long context, another cache-size's worth on top (512K expert slots 105 -> 955).  The engine
# derives the format from the file's own size, so no format variable is needed - but keep the Q4 path optional so
# a deployment that never ran tools/rdna2/q4_latent.py still works.
$q4 = 'E:/Edge0-120B-q3k/latent_dense_q4.bin'
if ($Dense -eq '') {
    $Dense = if (Test-Path $q4) { $q4 } else { 'E:/Edge0-120B-q3k/latent_dense_q8.bin' }
}
$env:STRATA_LATENT_DENSE = $Dense
if ($DenseFmt -ne '') { $env:STRATA_LATENT_DENSE_FMT = $DenseFmt } else { Remove-Item Env:\STRATA_LATENT_DENSE_FMT -ErrorAction SilentlyContinue }
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
