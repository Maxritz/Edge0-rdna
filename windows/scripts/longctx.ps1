$ErrorActionPreference = "Continue"
$bin   = "D:\edge0\wt\win\build-hip\bin"
$model35 = "C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
$model288 = "C:\x\qwen3.8-flash-next-reap-288-Q4_K_M.gguf"
$port = 8099

function RunArm($tag, $model, $ctx, $extra) {
    $log = "D:\edge0\lc-$tag.log"; $lerr = "$log.err"
    foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }
    # -ngl 999 lets common_fit_params REDUCE the layer count when the 256k/512k KV does not
    # fit in 12 GiB VRAM. Pinning -ngl 99 makes it abort: "n_gpu_layers already set by user".
    $a = @("-m", $model, "-ngl", "999", "-c", "$ctx", "-t", "32", "--no-webui", "--port", "$port",
           "--reasoning-format", "none") + $extra
    $p = Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
         -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 200; $i++) {
        Start-Sleep -Seconds 2
        if ($p.HasExited) { break }
        try { if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3).status -eq "ok") { $ok = $true; break } } catch {}
        $t = (Get-Content $lerr -Tail 1 -ErrorAction SilentlyContinue)
        if ($t -match "error|failed|assert") { Write-Output "$tag LOAD-ERR: $t"; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; return }
    }
    if (-not $ok) {
        Write-Output "$tag : LOAD-FAIL"
        Get-Content $lerr -Tail 3 -ErrorAction SilentlyContinue | ForEach-Object { Write-Output "    $_" }
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; return
    }
    # long prompt to actually populate the KV, then 32 decode tokens
    $body = @{ prompt = ("Summarize the history of computing in detail. " * 400) + " Now write a long essay.";
               n_predict = 32; temperature = 0.7; top_p = 0.80; top_k = 20; presence_penalty = 1.5 } |
           ConvertTo-Json -Compress
    try { Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 3600 | Out-Null }
    catch { Write-Output "$tag GENFAIL: $_" }
    Start-Sleep -Seconds 2
    Get-Content $lerr | Select-String "n_ctx|n_cpu_moe|CPU model size|offloaded|VRAM|prompt eval time|eval time =|KV self|buffer" |
        Select-Object -Last 8 | ForEach-Object { Write-Output "  $($_.Line.Trim())" }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
}

RunArm "35b-256k" $model35 262144 @()
RunArm "35b-512k" $model35 524288 @()
RunArm "35b-256k-kvq" $model35 262144 @("--cache-type-k","q8_0","--cache-type-v","q4_0")
RunArm "288-256k" $model288 262144 @("-cmoe")
