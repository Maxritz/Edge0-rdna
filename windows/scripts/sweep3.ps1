$ErrorActionPreference = "Continue"
$bin = "D:\edge0\wt\win\build-hip\bin"
$log = "D:\edge0\sweep3.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

# Hypothesis H1: thread oversubscription, not GPU placement, caused the 4.19/0.34 collapse.
# Sweep threads x ncmoe around the pp/tg peaks. Also an f16-KV control.
$a = @("-m","C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
       "-ngl","99","-ncmoe","1,2,3,6,20","-t","6,8,12,32",
       "-fa","auto","-ctk","q8_0","-ctv","q4_0",
       "-p","512","-n","128","-r","2","-o","csv")
$p = Start-Process -FilePath "$bin\llama-bench.exe" -ArgumentList $a `
     -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
$p | Wait-Process -Timeout 2400 -ErrorAction SilentlyContinue
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
Import-Csv $log -ErrorAction SilentlyContinue |
  ForEach-Object { [pscustomobject]@{
      ncmoe=$_.n_cpu_moe; threads=$_.n_threads; test=$_.test
      tps=[math]::Round([double]$_.avg_ts,2); sd=[math]::Round([double]$_.stddev_ts,2) } } |
  Sort-Object ncmoe,threads,test | Format-Table -AutoSize | Out-String -Width 160 | Write-Output
