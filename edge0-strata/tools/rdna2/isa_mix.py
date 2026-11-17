#!/usr/bin/env python3
"""Instruction-mix histogram for a gfx103x kernel dump, aimed at the "more work per instruction" question.

    llvm-objdump -d --mcpu=gfx1031 dev_34.elf > isa.txt
    python isa_mix.py isa.txt [--kernel SUBSTR]

Reports per kernel: total instructions, the top mnemonics, and the four counts that decide whether a kernel
is issue-bound rather than latency- or bandwidth-bound:

  VOPD      two VALU ops in ONE instruction word (RDNA2 dual issue).  If this is 0 on an ALU-bound kernel,
            the compiler is leaving up to 2x of the vector issue bandwidth on the table.
  v_pk_*    packed fp16 (or fp32x2) arithmetic: two lanes of math per instruction.
  v_dot*    dot products: v_dot2_f32_f16 (fp16 pairs), v_dot4_i32_i8 (dp4a), v_dot8_i32_i4 (int4 octets).
  s_*       the scalar unit.  It issues in PARALLEL with the VALU, so index arithmetic there is free; a hot
            loop whose addresses live in VGPRs is paying vector issue slots for bookkeeping.
"""
import collections
import re
import sys


def parse(path):
    cur = None
    instrs = collections.defaultdict(list)
    raw = open(path, "rb").read()
    # PowerShell's ">" writes UTF-16LE; cmd's writes UTF-8.  Sniff instead of guessing at the call site.
    text = raw.decode("utf-16-le", errors="replace") if b"\x00" in raw[:400] else raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        m = re.match(r"^\s*([0-9a-f]{16}) <(.+)>:", line)
        if m:
            cur = m.group(2)
            continue
        body = line.split("//")[0].strip()
        if not body or cur is None:
            continue
        op = body.split()[0].split("@")[0]
        instrs[cur].append(op)
    return instrs


def demangle(name):
    """Just enough Itanium to read AMDGPU kernel names: drop _ZN... length prefixes."""
    if not name.startswith("_ZN"):
        return name
    parts, i = [], 3
    while i < len(name) and name[i] != "E":
        j = i
        while name[j].isdigit():
            j += 1
        if j == i:
            break
        n = int(name[i:j])
        parts.append(name[j:j + n])
        i = j + n
    return "::".join(parts)


def main():
    path = sys.argv[1]
    want = None
    if "--kernel" in sys.argv:
        want = sys.argv[sys.argv.index("--kernel") + 1]
    instrs = parse(path)
    for name, ops in instrs.items():
        pretty = demangle(name)
        if want and want not in pretty and want not in name:
            continue
        if not ops:
            continue
        c = collections.Counter(ops)
        total = len(ops)
        vopd = sum(n for o, n in c.items() if o.startswith("vopd"))
        pk = sum(n for o, n in c.items() if o.startswith("v_pk_") or o.startswith("v_dual"))
        dot = sum(n for o, n in c.items() if o.startswith("v_dot"))
        scalar = sum(n for o, n in c.items() if o.startswith("s_"))
        mem = sum(n for o, n in c.items() if o.startswith(("global_", "flat_", "buffer_", "ds_", "scratch_")))
        valu = sum(n for o, n in c.items() if o.startswith("v_") and not o.startswith("vopd"))
        print(f"\n== {pretty}   ({total} instructions)")
        print(f"   VOPD (dual-issue words) {vopd:5d}   packed v_pk_/v_dual {pk:5d}   v_dot* {dot:5d}")
        print(f"   VALU {valu:5d}   scalar s_* {scalar:5d}   memory {mem:5d}")
        for op, n in c.most_common(10):
            print(f"   {n:5d}  {op}")


if __name__ == "__main__":
    main()
