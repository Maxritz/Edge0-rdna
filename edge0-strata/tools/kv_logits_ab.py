#!/usr/bin/env python3
"""A/B the KV cache format on the PROMPT'S OWN LOGITS, with no decoding in the way.

Why this exists: comparing replies (`serve/` + greedy) cannot separate a KV format's error from the decoder's own
noise - the same format gave two different greedy replies for the same 2 512-token prompt.  This harness asks the
engine for the logits at the last prompt position (`--dump-logits` with `--max-new 1`, so the dump holds the
decision point and nothing is sampled), runs the same tokens under each `--kv` format, and reports how far the
top-1 margin moves.

    python tools/kv_logits_ab.py --config serve/config-rdna2.json --formats int8 q4_0 --repeats 2

Every run is a fresh engine process, so the conversation cache and the expert-cache history cannot differ between
arms.  The dump format is the engine's: int32 n_vocab, int32 n_rows, then n_rows x n_vocab float32.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

DEFAULT_TEXT = " ".join(f"User {i} asks a question about the model." for i in range(1, 61))


def tokenize(cfg, text):
    """The config's tokenizer, the same one the serve stack uses."""
    import strata_tokenizer as ST

    tdir = Path(cfg["tokenizer"])
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    try:
        tcfg = json.loads((tdir / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tcfg = {}
    tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
    return list(tok.encode(text)), tokens


def run_arm(cfg, fmt, ids, workdir):
    """One engine run: the same tokens under `fmt`, dumping the last prompt position's logits.

    A native (IQ) pack does not reach `--dump-logits`' write site - it empties the file - so it uses the engine's
    native equivalent, `STRATA_DUMP_FIRST_LOGITS`: the verifier's row 0 of the first window, which is the prompt's last
    position, written as `n_vocab` raw float32 with no header or sampling in the way.
    """
    args = [str(a) for a in cfg["args"] if a != "--serve"]
    native = "--native" in args
    dump = Path(workdir) / f"logits-{fmt}-{len(list(Path(workdir).iterdir()))}.bin"
    args += ["--kv", fmt, "--tokens", ",".join(str(i) for i in ids), "--max-new", "1"]
    if native:
        env_extra = {"STRATA_DUMP_FIRST_LOGITS": str(dump)}
    else:
        env_extra = {}
        args += ["--dump-logits", str(dump), "--logits-stride", "1000000000"]
    env = dict(os.environ)          # the config's env is what the serve stack hands the engine (latent layout, PLE...)
    env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
    env.update(env_extra)
    proc = subprocess.run([cfg["exe"]] + args, cwd=cfg.get("cwd") or None, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not dump.exists():
        raise SystemExit(f"[{fmt}] engine failed (exit {proc.returncode}):\n{(proc.stderr or proc.stdout)[-2000:]}")
    return dump


def read_last_row(path, native=False):
    import struct

    raw = Path(path).read_bytes()
    if raw[:4] == b"NLG1":            # STRATA_NATIVE_LOGITS: walk the streamed rows, keep the last
        n_vocab, _reserved = struct.unpack_from("<ii", raw, 4)
        off, last = 8, None
        while off + 8 + n_vocab * 4 <= len(raw):
            pos, nv = struct.unpack_from("<ii", raw, off)
            if nv != n_vocab:
                break
            row = struct.unpack_from(f"<{n_vocab}f", raw, off + 8)
            last = (pos, row)
            off += 8 + n_vocab * 4
        if last is None:
            raise SystemExit(f"{path}: NLG1 stream holds no complete row ({len(raw)} bytes)")
        return n_vocab, 1, last[1]
    if native:            # STRATA_DUMP_FIRST_LOGITS: one bare row, no header
        if len(raw) < 4 or len(raw) % 4 != 0:
            raise SystemExit(f"{path}: not a whole float32 row ({len(raw)} bytes)")
        n_vocab = len(raw) // 4
        return n_vocab, 1, struct.unpack_from(f"<{n_vocab}f", raw, 0)
    n_vocab, n_rows = struct.unpack_from("<ii", raw, 0)
    row_bytes = n_vocab * 4
    if n_rows < 1 or len(raw) < 8 + n_rows * row_bytes:
        raise SystemExit(f"{path}: header says {n_vocab} x {n_rows}, file is {len(raw)} bytes")
    off = 8 + (n_rows - 1) * row_bytes          # the last row is the full prompt's decision point
    return n_vocab, n_rows, struct.unpack_from(f"<{n_vocab}f", raw, off)


def top2(row):
    """(top1 id, top1 logit, top2 id, top2 logit) without numpy."""
    best = second = float("-inf")
    bi = si = -1
    for i, v in enumerate(row):
        if v > best:
            second, si, best, bi = best, bi, v, i
        elif v > second:
            second, si = v, i
    return bi, best, si, second


def measure(cfg, ids, vocab, formats, repeats, workdir):
    """All arms for one prompt: the same token ids under every format, `repeats` fresh processes each."""
    native = "--native" in [str(x) for x in cfg["args"]]
    arms = {}
    for fmt in formats:
        rows = []
        for rep in range(repeats):
            dump = run_arm(cfg, fmt, ids, workdir)
            n_vocab, n_rows, row = read_last_row(dump, native=native)
            bi, bv, si, sv = top2(row)
            rows.append({"n_vocab": n_vocab, "n_rows": n_rows, "top1": bi, "top1_logit": bv,
                         "top2": si, "top2_logit": sv, "margin": bv - sv,
                         "top1_text": vocab[bi] if bi < len(vocab) else None})
            print(f"[{len(ids)} tok {fmt} rep{rep}] top1 {bi} ({ascii(rows[-1]['top1_text'])}) "
                  f"logit {bv:.6f}  margin {bv - sv:.6f}", flush=True)   # ascii(): cp1252 stdout cannot print '\u0120'
        arms[fmt] = rows
    return arms


def summarise(arms, formats):
    """Report the within-format repeatability and the cross-format deltas for one prompt."""
    first = {f: arms[f][0] for f in formats}
    out = {"repeatable": {}, "cross": {}}
    for fmt in formats:
        reps = arms[fmt]
        out["repeatable"][fmt] = {
            "top1": all(r["top1"] == reps[0]["top1"] for r in reps),
            "margin_spread": max(r["margin"] for r in reps) - min(r["margin"] for r in reps),
        }
        print(f"{fmt}: repeatable top1 {out['repeatable'][fmt]['top1']}, "
              f"margin spread {out['repeatable'][fmt]['margin_spread']:.6f}")
    ref = formats[0]
    for fmt in formats[1:]:
        x, y = first[ref], first[fmt]
        out["cross"][f"{ref}_vs_{fmt}"] = {
            "same_argmax": x["top1"] == y["top1"],
            "margin_delta": abs(x["margin"] - y["margin"]),
            "top1_logit_delta": abs(x["top1_logit"] - y["top1_logit"]),
        }
        print(f"{ref} vs {fmt}: same argmax {out['cross'][f'{ref}_vs_{fmt}']['same_argmax']}, "
              f"margin {x['margin']:.6f} vs {y['margin']:.6f} "
              f"(delta {out['cross'][f'{ref}_vs_{fmt}']['margin_delta']:.6f}), "
              f"top1 logit delta {out['cross'][f'{ref}_vs_{fmt}']['top1_logit_delta']:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="serve/config-rdna2.json")
    ap.add_argument("--formats", nargs="+", default=["int8", "q4_0"])
    ap.add_argument("--repeats", type=int, default=2, help="engine runs per format (repeatability of the logits)")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--engine-arg", action="append", default=[],
                    help="extra engine argument, repeat for a flag with its value (e.g. --engine-arg --adapt-every --engine-arg 0)")
    ap.add_argument("--lengths", nargs="+", type=int,
                    help="prompt lengths to sweep: tokenize the filler once and take the first N ids of each")
    ap.add_argument("--out", default="kv_logits_ab.json")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    cfg["args"] = [str(x) for x in cfg["args"]] + [str(x) for x in a.engine_arg]
    if a.lengths:            # one filler long enough for the longest arm, so every length is a prefix of the same text
        filler = " ".join(f"Paragraph {i} discusses a different topic in ordinary words." for i in range(1, 400))
        ids_all, vocab = tokenize(cfg, filler)
        if len(ids_all) < max(a.lengths):
            raise SystemExit(f"filler reached only {len(ids_all)} tokens, wanted {max(a.lengths)}")
    else:
        ids_all, vocab = tokenize(cfg, a.text)

    prompts = [ids_all[:n] for n in a.lengths] if a.lengths else [ids_all]
    report = {"config": a.config, "lengths": [len(p) for p in prompts], "runs": {}}
    with tempfile.TemporaryDirectory() as workdir:
        for ids in prompts:
            print(f"\n== prompt: {len(ids)} tokens ==", flush=True)
            arms = measure(cfg, ids, vocab, a.formats, a.repeats, workdir)
            report["runs"][str(len(ids))] = {"arms": arms, "summary": summarise(arms, a.formats)}

    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
