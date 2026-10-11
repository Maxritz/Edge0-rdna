$p = Get-Process llama-bench -ErrorAction SilentlyContinue
Write-Output "RUNNING=$($null -ne $p)"
Write-Output '---LOG---'
Get-Content D:\edge0\sweep.log -ErrorAction SilentlyContinue | Select-Object -Last 30
