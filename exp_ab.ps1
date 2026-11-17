# Reusable alternating-pair A/B harness for Edge0 RDNA2 experiments.
#   powershell -File exp_ab.ps1 -Tag base -Reps 2
#   powershell -File exp_ab.ps1 -Tag opt4 -Reps 2 -Env "STRATA_X=1"
# Runs baseline (A) and variant (B) alternately: A1 B1 A2 B2, so batch drift is shared.
# Each run: canonical 4k prefill + 8 decode, full trace, token output captured for the byte-exact gate.
param(
    [string]$Tag = 'ab',
    [int]$Reps = 2,
    [string]$Env = '',       # semicolon-separated NAME=VALUE applied to B arm only
    [string]$AddArgs = '',   # extra CLI args applied to B arm only
    [string]$Tokens = 'prompt4k.txt',
    [string]$Prefill = '832',
    [int]$MaxContext = 4096,
    [int]$MaxNew = 8,
    [string]$Pack = ''       # B arm only: alternate pack dir; overrides --pack and STRATA_LATENT_A
)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\rr\Desktop\Edge0-rdna'
$build = Join-Path $root 'edge0-strata\build'
$env:PATH = "C:\Rocm10\bin;" + $env:PATH

# Common latent-path environment for BOTH arms.
$common = @{
    STRATA_LATENT_A = 'E:/Edge0-120B-q3k/experts.bin'
    STRATA_LATENT_LABELS = 'E:/Edge0-120B-q3k/labels.bin'
    STRATA_LATENT_DENSE = 'E:/Edge0-120B-q3k/latent_dense_q4.bin'
    STRATA_LATENT_DENSE_FMT = ''
    STRATA_EMBED_VRAM = '1'
    STRATA_LATENT_FUSED = '1'
    STRATA_LATENT_FUSED_DU = '1'
    STRATA_LATENT_U_GPU = '1'
    STRATA_LAT_FUSED_Q3K = '1'
    STRATA_MTP_SIDE_STREAM = '1'
    STRATA_PREFILL_TIMING = '1'
}
$baseArgs = @(
    '--pack', 'E:\Edge0-120B-q3k',
    '--native', 'E:\Edge0-120B-q3k\model-skeleton.gguf',
    '--native-dense-gguf', 'E:\Edge0-120B-q3k\model-skeleton.gguf',
    '--ple-gguf', 'E:\Edge0-120B-q3k\ple-fp8.gguf',
    '--ple-io', 'direct',
    '--expert-profile', 'E:\Edge0-120B-q3k\profile.bin',
    '--expert-cache', 'auto',
    '--prefill', $Prefill,
    '--spec', '4', '--spec-min-p', '0.5',
    '--mtp', 'E:\Edge0-120B-q3k\mtp',
    '--tokens-file', (Join-Path $root $Tokens),
    '--max-context', "$MaxContext",
    '--kv', 'int8', '--pcie-frac', '0.1',
    '--gpu', '0', '--max-new', "$MaxNew", '--stats',
    '--vram-reserve-mib', '512'
)

# Parse variant env into a hashtable. Variant keys that are not common keys are cleared
# before every arm so a B-only setting can never leak into the following A run.
$var = @{}
if ($Env -ne '') { foreach ($pair in $Env.Split(';')) { $kv = $pair.Split('=', 2); if ($kv.Count -eq 2) { $var[$kv[0]] = $kv[1] } } }
$varOnly = @($var.Keys | Where-Object { -not $common.ContainsKey($_) })

function Invoke-Arm([string]$arm, [string]$name, [hashtable]$variant, [string]$extra, [string]$packDir) {
    # Reset: common env for both arms, then remove any variant-only keys from a prior run.
    foreach ($k in $common.Keys) {
        if ($common[$k] -eq '') { Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
        else { Set-Item "Env:\$k" $common[$k] }
    }
    foreach ($k in $varOnly) { Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
    foreach ($k in $variant.Keys) { Set-Item "Env:\$k" $variant[$k] }
    $runArgs = @($baseArgs)
    if ($packDir -ne '') {
        # Only the expert blob and the --pack manifest change; labels/dense/mtp/native stay q3k.
        Set-Item "Env:\STRATA_LATENT_A" ($packDir + '/experts.bin')
        for ($j = 0; $j -lt $runArgs.Count - 1; $j++) {
            if ($runArgs[$j] -eq '--pack') { $runArgs[$j + 1] = $packDir; break }
        }
    }
    $log = Join-Path $root "ab_${Tag}_${name}.log"
    $runArgs = $runArgs + ($extra -split '\s+' | Where-Object { $_ -ne '' })
    $p = Start-Process -FilePath (Join-Path $build 'strata.exe') -ArgumentList $runArgs -WorkingDirectory $build `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
    Get-Content "$log.err" | Set-Content $log
    Get-Content "$log.out" | Add-Content $log
    Remove-Item "$log.out", "$log.err" -ErrorAction SilentlyContinue
    $t = ((Get-Content $log) -join ' ') -replace '\s+', ' '
    $prefill = [regex]::Match($t, 'prefill \d+ tokens in [\d.]+ ms -> ([\d.]+) tok/s').Groups[1].Value
    $decode = [regex]::Match($t, 'decode \d+ tokens in [\d.]+ ms -> ([\d.]+) tok/s').Groups[1].Value
    $blobs = [regex]::Match($t, 'streamed (\d+) blobs').Groups[1].Value
    $gate = [regex]::Match($t, 'gemm gate/up (\d+) (\d+)').Groups[2].Value
    $down = [regex]::Match($t, 'gemm down (\d+) (\d+)').Groups[2].Value
    $tok = ((Get-Content $log | Where-Object { $_ -match '^\s*output\s*:' }) -replace '^\s*output\s*:\s*','').Trim()
    [pscustomobject]@{ arm=$arm; name=$name; exit=$p.ExitCode; prefill=$prefill; decode=$decode; blobs=$blobs; gate=$gate; down=$down; tokens=$tok; log=$log }
}

$rows = @()
for ($i = 1; $i -le $Reps; $i++) {
    $rows += Invoke-Arm 'A' "base$i" @{} '' ''
    $rows += Invoke-Arm 'B' "var$i"  $var $AddArgs $Pack
}
$rows | Format-Table arm,name,exit,prefill,decode,blobs,gate,down -AutoSize | Out-String -Width 160 | Write-Output
# Per-arm spread: the standing rule forbids accepting a delta inside the noise.
$stats = foreach ($a in @('A','B')) {
    $p = @($rows | Where-Object { $_.arm -eq $a -and $_.prefill -ne '' } | ForEach-Object { [double]$_.prefill })
    $d = @($rows | Where-Object { $_.arm -eq $a -and $_.decode -ne '' } | ForEach-Object { [double]$_.decode })
    if ($p.Count -eq 0) { continue }
    $pm = ($p | Measure-Object -Average -Minimum -Maximum)
    $dm = ($d | Measure-Object -Average -Minimum -Maximum)
    [pscustomobject]@{ arm=$a; n=$p.Count; prefill_mean=[math]::Round($pm.Average,2); prefill_min=$pm.Minimum; prefill_max=$pm.Maximum; prefill_spread_pct=[math]::Round(100*($pm.Maximum-$pm.Minimum)/$pm.Average,1); decode_mean=[math]::Round($dm.Average,2); decode_spread_pct=[math]::Round(100*($dm.Maximum-$dm.Minimum)/$dm.Average,1) }
}
$stats | Format-Table -AutoSize | Out-String -Width 140 | Write-Output
# Correctness gate: all arms must have identical token strings.
$uniq = @($rows | Where-Object { $_.tokens -ne '' } | Select-Object -ExpandProperty tokens -Unique)
Write-Output "TOKEN_VARIANTS=$($uniq.Count)"
if ($uniq.Count -gt 0) { Write-Output "TOKENS=$($uniq[0])" }
$rows | Export-Csv (Join-Path $root "ab_${Tag}.csv") -NoTypeInformation
