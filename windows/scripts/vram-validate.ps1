$ErrorActionPreference = "Continue"
$bin  = "D:\edge0\wt\win\build-hip\bin"
$model = "C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
$port = 8087

function RunArm($tag, $extra, $envs) {
    $log  = "D:\edge0\v-$tag.log"
    $lerr = "$log.err"
    if (Test-Path $log)  { Remove-Item $log }
    if (Test-Path $lerr) { Remove-Item $lerr }
    foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
    $a = @("-m", $model, "-ngl", "99", "--n-cpu-moe", "20", "-fa", "auto",
           "--ctx-size", "4096", "--no-webui", "--port", "$port") + $extra
    $p = Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
         -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 150; $i++) {
        Start-Sleep -Seconds 2
        if ($p.HasExited) { break }
        try { $h = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3
              if ($h.status -eq "ok") { $ok = $true; break } } catch {}
    }
    if (-not $ok) {
        Write-Output "$tag : LOAD-FAIL :: $((Get-Content $lerr -Tail 3 -ErrorAction SilentlyContinue) -join ' | ')"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        foreach ($k in $envs.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
        return
    }
    $body = @{ prompt = "Write a detailed essay about the history of computing."; n_predict = 128; temperature = 0.2 } | ConvertTo-Json -Compress
    try { Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1200 | Out-Null }
    catch { Write-Output "$tag : GENFAIL $_" }
    Start-Sleep -Seconds 2
    $ev = (Get-Content $lerr | Select-String "eval time =" | Select-Object -Last 1).Line
    $vi = (Get-Content $lerr | Select-String "vram-trace. INIT" | Select-Object -Last 1).Line
    $vs = (Get-Content $lerr | Select-String "vram-trace. FINAL" | Select-Object -Last 1).Line
    $vf = (Get-Content $lerr | Select-String "vram-trace" | Where-Object { $_ -match "off|failed" } | Select-Object -First 1).Line
    $pt = (Get-Content $lerr | Select-String "POOL2 init|pref-trace" | Select-Object -Last 1).Line
    Write-Output "=== $tag"
    if ($vi) { Write-Output "  init: $($vi.Trim())" }   else { Write-Output "  init: (none)" }
    if ($ev) { Write-Output "  eval: $($ev.Trim())" }    else { Write-Output "  eval: (none)" }
    if ($vs) { Write-Output "  finl: $($vs.Trim())" }
    if ($vf) { Write-Output "  off : $($vf.Trim())" }
    if ($pt) { Write-Output "  pref: $($pt.Trim())" }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    foreach ($k in $envs.Keys) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 4
}

# A: baseline - no RAM pool, no VRAM tier
RunArm "plain" @() @{}

# B: RAM L1 warm pool only (the pre-existing warm tier)
RunArm "rampool" @("--pool-mb","2048") @{ "E0_PREF_TRACE_N"="24" }

# C: VRAM hot tier (the new T1/T2 work) - small, auto-shrinks to free VRAM
RunArm "vram" @() @{ "E0_VRAM_MB"="1024"; "E0_VRAM_TRACE_N"="8"; "E0_VRAM_MAX_SWAPS"="4" }

# D: both tiers - RAM warm feeding the VRAM hot tier
RunArm "both" @("--pool-mb","2048") @{ "E0_VRAM_MB"="1024"; "E0_VRAM_TRACE_N"="8"; "E0_VRAM_MAX_SWAPS"="4"; "E0_PREF_TRACE_N"="24" }
