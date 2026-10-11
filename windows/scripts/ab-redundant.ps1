$ErrorActionPreference = "Continue"
$bin   = "D:\edge0\wt\win\build-hip\bin"
$model = "C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
$log = "D:\edge0\ab-redundant.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

$body = @{ prompt = "Write a detailed essay about the history of computing."; n_predict = 48;
           temperature = 0.7; top_p = 0.80; top_k = 20; presence_penalty = 1.5 } | ConvertTo-Json -Compress

function RunArm($tag, $envs) {
    $port = 8096
    if (Test-Path $log)  { Remove-Item $log }
    if (Test-Path $lerr) { Remove-Item $lerr }
    foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
    $a = @("-m", $model, "-ngl", "99", "--n-cpu-moe", "20", "-fa", "auto",
           "--ctx-size", "4096", "--no-webui", "--port", "$port",
           "--reasoning-format", "none")
    $p = Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
         -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 150; $i++) {
        Start-Sleep -Seconds 2
        if ($p.HasExited) { break }
        try { if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3).status -eq "ok") { $ok = $true; break } } catch {}
    }
    if (-not $ok) { Write-Output "$tag : LOAD-FAIL"; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; foreach ($k in $envs.Keys) { Remove-Item "env:$k" }; return }
    try { Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 900 | Out-Null } catch { Write-Output "$tag GENFAIL" }
    Start-Sleep -Seconds 2
    $ev = (Get-Content $lerr | Select-String "eval time =" | Select-Object -Last 1).Line
    $vt = (Get-Content $lerr | Select-String "vram-trace. FINAL|vram-trace. step" | Select-Object -Last 1).Line
    $in = (Get-Content $lerr | Select-String "vram-trace. INIT" | Select-Object -Last 1).Line
    if ($in)  { Write-Output "$tag INIT: $($in.Trim())" }
    if ($ev)  { Write-Output "$tag EVAL: $($ev.Trim())" }
    if ($vt)  { Write-Output "$tag TRACE: $($vt.Trim())" }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 4
}

# A: VRAM tier OFF (baseline)
RunArm "POOL-OFF" @{}
# B: VRAM tier ON - if src0 is already VRAM-resident this must NOT beat A
RunArm "POOL-ON"  @{ "E0_VRAM_MB"="1024"; "E0_VRAM_TRACE_N"="1"; "E0_VRAM_MAX_SWAPS"="8" }
