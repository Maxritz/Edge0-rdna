#!/usr/bin/env bash
# Which kernel reads the Q3_K gate/up expert bytes during prefill on gfx1031?
# One real prefill through the engine with the HIP runtime's API log at AMD_LOG_LEVEL=4 (it names every dispatched
# kernel), then the kernels seen, with counts.
# NOTE: paths are project-local on purpose - the Windows python does not share Git Bash's /tmp.
set -u
cd /c/Users/rr/Desktop/Edge0-rdna/edge0-strata

LOG=pf_trace.log
IN=pf_in.json
OUT=pf_out.json
: > "$LOG"

PID=$(netstat -ano 2>/dev/null | grep ":8130" | grep -i listen | awk '{print $5}' | head -1)
[ -n "${PID:-}" ] && taskkill //PID "$PID" //F >/dev/null 2>&1 && sleep 2

# a ~1,450-token prompt inside --max-context 2048, and a 1-token answer: the run is the prompt read
.venv/Scripts/python.exe - <<'PY'
import json
filler = ("The Radeon RX 6700 XT is an RDNA2 part, gfx1031, with twelve gigabytes of memory. " * 12)
json.dump({"model": "edge0_120b", "temperature": 0, "max_tokens": 1,
           "messages": [{"role": "user", "content": filler + "\nIn one word: what architecture is it?"}]},
          open("pf_in.json", "w"))
PY

AMD_LOG_LEVEL=4 .venv/Scripts/python.exe serve/server.py --engine strata \
  --config serve/config-rdna2.json --port 8130 >> "$LOG" 2>&1 &
SPID=$!

ready=0
for i in $(seq 1 90); do
  if curl -s -o /dev/null --max-time 2 http://127.0.0.1:8130/v1/models; then ready=1; break; fi
  sleep 2
done
echo "server ready: $ready"

if [ "$ready" = 1 ]; then
  curl -s --max-time 220 -X POST http://127.0.0.1:8130/v1/chat/completions \
    -H "Content-Type: application/json" -d @"$IN" -o "$OUT"
  echo "request exit: $?"
else
  echo "server never answered"; tail -20 "$LOG"
fi

sleep 2
kill "$SPID" >/dev/null 2>&1
sleep 3
for p in $(netstat -ano 2>/dev/null | grep ":8130" | grep -i listen | awk '{print $5}'); do taskkill //PID "$p" //F >/dev/null 2>&1; done

echo "=== trace: $(wc -l < "$LOG") lines ==="
echo "=== distinct device kernels the trace names, by number of occurrences ==="
grep -aoE "_Z[0-9A-Za-z_]{6,}" "$LOG" | sort | uniq -c | sort -rn | head -25
echo "=== raw trace lines around a named kernel (the format) ==="
grep -a "_Z[0-9A-Za-z_]\{6,\}" "$LOG" | head -4 | cut -c1-180
echo "=== which Q3_K gate/up candidates appear ==="
for n in native_q3_k_mmvq_kernel launch_multi experts_latent_q3k lat_q3k copy16 gather_native mmvq_ kernel; do
  printf "  %-26s %s\n" "$n" "$(grep -ac "$n" "$LOG")"
done
