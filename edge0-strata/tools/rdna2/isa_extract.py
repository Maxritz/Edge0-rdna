#!/usr/bin/env python3
"""Extract the device ELFs from strata.exe's .hip_fat section and name them by arch.

COFF section names are 8 bytes, so `.hip_fatbin` arrives as `.hip_fat`; llvm-objcopy refuses the long name.
The blob is a clang offload bundle: a header, then one ELF per target arch, each with its section header
table at its END.  So a candidate's true extent runs to the next VALID ELF header, not to a computed size
(the first attempt sliced at e_shoff/e_shnum read from the wrong field and cut every ELF short).
"""
import struct
import sys

WANT = [b"latent_u_tile_k", b"latent_u_tile_s_k", b"latent_d_tile", b"latent_d_fast_k",
        b"dot2", b"lat_q3k", b"latent_dense_kernel", b"gemm"]


def valid_header(d, o):
    """A real ELF64/ELF32 header, not a stray magic inside a string table."""
    if o + 64 > len(d):
        return False
    ei_class, ei_data, ei_version, ei_osabi = d[o + 4], d[o + 5], d[o + 6], d[o + 7]
    # 0 = SYSV, 3 = Linux, 64 = ELFOSABI_AMDGPU_HSA: the device ELFs this targets use 64.
    if ei_class not in (1, 2) or ei_data != 1 or ei_version != 1 or ei_osabi not in (0, 3, 64, 65):
        return False
    if ei_class == 2:
        e_type, e_machine = struct.unpack_from("<H", d, o + 0x10)[0], struct.unpack_from("<H", d, o + 0x12)[0]
    else:
        e_type, e_machine = struct.unpack_from("<H", d, o + 0x10)[0], struct.unpack_from("<H", d, o + 0x12)[0]
    return e_type in (1, 2, 3) and e_machine == 224          # ET_*, EM_AMDGPU


def main():
    d = open(sys.argv[1], "rb").read()
    offs, i = [], 0
    while True:
        i = d.find(b"\x7fELF", i)
        if i < 0:
            break
        if valid_header(d, i):
            offs.append(i)
        i += 4
    print(f"{sys.argv[1]}: {len(d)} bytes, {len(offs)} valid AMDGPU ELFs")
    for n, o in enumerate(offs):
        end = offs[n + 1] if n + 1 < len(offs) else len(d)
        blob = d[o:end]
        # the arch name sits in the ELF's own .rodata/strtab as "gfx1031\0"; report it so the dump is identifiable
        arch = ""
        for a in (b"gfx1031", b"gfx1030", b"gfx1100", b"gfx90a"):
            if a in blob:
                arch = a.decode()
                break
        hits = sorted({x.decode() for x in WANT if x in blob})
        fn = f"dev_{n}.elf"
        open(fn, "wb").write(blob)
        print(f"  {fn}: {len(blob):>9} B  arch={arch or '?':8s} {hits if hits else ''}")


if __name__ == "__main__":
    main()
