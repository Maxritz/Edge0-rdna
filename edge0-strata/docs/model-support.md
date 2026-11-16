# Which models this tree can run, and what adding another one takes

Verified on this machine (RDNA2 / gfx1031, Windows, HIP) on 2026-11-15.

## The three engine paths

| Model family | GGUF `general.architecture` | Engine | Binary | Status here |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (and `edge0` packs) | `qwen4exp`, `edge0` | `src/program` (main path) | `strata.exe` | **Running** — the 120B q3 pack in `E:/Edge0-120B-q3k` |
| Qwen3.6-35B-A3B | `qwen35moe` | `src/qwen35` | `strata.exe` (family switch, `STRATA_HAVE_QWEN35=1`) or `strata-q36.exe` | **Compiled in**; needs a `qwen35moe` GGUF |
| DeepSeek-V4.1-Flash | a `maya-ds4` pack | `src/dsv41` | `strata-dsv41.exe` | **Built today** with `tools/hip/build_dsv41.bat` (`strata-dsv41 0.1.55 build 8c3317e88`) |

There is no fourth path. Every other architecture string is refused by name, and the refusal is the design:
`core::compiled_geometry_mismatch` compares 19 geometry fields against `core::kCompiledGeometry` (Flash-Next's
shape — 2560 wide, 48 layers, `full_attention_interval` 4, head_dim 256, 24 heads / 2 KV, hyper-connections 4 x
320, indexer 4 x 128, FF 640), because the kernels are compiled for those numbers.

### What that gate means in practice

Runnable today **without a new engine**: Flash-Next fine-tunes, **pruned-expert variants** (`n_expert` is run-time;
the Coder ships fewer than 512), different vocab sizes, and any quant type the pack loader reads. Everything else —
another layer count, another width, dense attention, no GDN — needs its own engine path, which is exactly why
Qwen3.6 and DeepSeek each got one instead of widening this gate.

## Running each path

```bat
:: Flash-Next / edge0 (the pack in serve/config-rdna2.json)
.strata.exe --pack E:/Edge0-120B-q3k --serve

:: Qwen3.6-35B-A3B: the same binary, switched on the file's architecture string.
:: --native takes the GGUF shard; the switch reads general.architecture before any other state exists.
strata.exe --native E:/models/qwen36-35b/qwen35moe-Q4_K_M-00001-of-00002.gguf --serve

:: DeepSeek-V4.1-Flash (its own executable, its own pack layout)
build-hip-rdna2/strata-dsv41.exe --pack <model>/maya-ds4 --serve
```

`strata-dsv41 --pack-info` prints what a pack declares; `--vision on|off` and `--tp N` are that engine's own.

### The Qwen3.6 serve config

`serve/config-qwen36.json` is the whole server stack pointed at a `qwen35moe` file:

```bat
python -m serve.server --engine strata --config serve/config-qwen36.json --port 8130
```

Every start prints the path it will take, before anything loads:

```
[strata] engine path: Qwen3.6-35B-A3B (src/qwen35) - model E:/models/qwen36-35b/model.gguf, general.architecture = qwen35moe
[strata] engine path: Flash-Next (src/program) - model E:/Edge0-120B-q3k/model-skeleton.gguf, general.architecture = qwen4exp
[strata] engine path: Flash-Next (src/program) - model x.gguf, general.architecture = llama - no engine path for that
         architecture in this build: the file will be refused by name
```

The architecture comes from the GGUF header (`general.architecture`, a few KB read), so it describes the file rather
than the config's hopes; a config whose `"arch"` disagrees with the file says so, and a file that cannot be read falls
back to the config's `"arch"`.

It names `E:/models/qwen36-35b/model.gguf` and `E:/models/qwen36-35b/tokenizer` as the place to put the model;
edit `"args"`/`"tokenizer"` if yours lives elsewhere. `--model` (or `--native`) is what triggers the family switch in
`strata.exe`: verified end to end with a two-key stub whose `general.architecture` is `qwen35moe` — the engine hands
the file to `strata-q36`, which then asks for `qwen35moe.attention.head_count`, i.e. it is reading it as Qwen3.6 and
not as Flash-Next.

**Missing files are reported before anything loads.** `serve/server.py` checks the model paths in a config's `args`
(`--pack`, `--native`, `--model`, `--ple-gguf`, `--native-dense-gguf`) at startup, so a config whose model is not on
disk stops in a second with the flag and the path named, instead of after minutes of loading with the engine's own
message buried in the log:

```
server.py: error: config-qwen36.json: the engine cannot start, these model paths do not exist:
  --model  E:/models/qwen36-35b/model.gguf
put the model there, or point "args" at the files you have (the model itself is the --native/--model value: a GGUF
shard, or for a pack the --pack directory)
```

## How to tell whether a model file can run: the gate, field by field

```
1. general.architecture -> ModelFamily          qwen4exp | edge0 | qwen35moe | (anything else: refused by name)
2. that family's guard                          qwen35moe: every key required; qwen4exp: six required, rest default
3. geometry_from_gguf                           reads "<arch>.*" keys; checks the redundant pairs agree
                                                (key vs value width, interval vs compress_ratios, FF vs shexp)
4. compiled_geometry_mismatch                   19 fields against kCompiledGeometry — the engine runs ONE shape
5. check_gguf_layout                            every table tensor present, shape-exact, inside its shard
```

`strata-gguf <file> --check` already prints the architecture guard for a single shard, and it is the first thing to
run on an unfamiliar file. Build it into the same directory as the engine:

```bat
tools\hip\build_dsv41.bat strata-gguf
build-hip-rdna2\strata-gguf.exe E:/Edge0-120B-q3k/model-skeleton.gguf --check
  -> version 3   tensors 1079   metadata 57   data_start 11012448   size 5991569248
     blk.* tensors 1074   globals 5
     Q2_0 0   BF16 484   F32 292
     geometry bracket: 0 out of range, 0 types unknown
     architecture guard: PASS
```

## Adding a fourth family — the real cost

1. **`src/artifact/model_family.hpp`**: enum value, `model_family_from_arch` alias, title, `model_family_runs`.
2. **`include/strata/core/family_layout.hpp`**: the geometry keys it reads and its **complete tensor table**
   (`layer_tensors` / `global_tensors`); that table is what turns "missing blk.7.attn_q.weight" into a good error.
3. **A guard** like `check_architecture_qwen35moe` if the file must carry every key.
4. **An engine path.** This is the whole job: the layer pattern (GDN + gated attention here, dense attention there,
   MLA for DeepSeek), the expert layout, the KV format and the sampler are all per-family. Qwen3.6's path is
   `src/qwen35` (≈ engine + kernels + reference), DeepSeek's is `src/dsv41` (vendored from Project Maya, MIT).
5. **`compiled_geometry_mismatch`**: add the fields whose values the new kernels bake in, so a wrong-size file is
   refused with the field name instead of mis-run.
6. **Tests that build a synthetic GGUF** (`tests/core/model_family_test.cpp` style) for accept/refuse, because a real
   model of the new family may not be on disk.

Nothing about a new family is cheap, but steps 1-3 and 5 are a day's work each and make the file *recognised* and
precisely refused; step 4 is the weeks-long part. That ordering — reader first, engine second — is why
`model_family_not_supported()` exists: recognising a family you cannot run still gives the user a correct message
instead of "not a model family Strata knows".
