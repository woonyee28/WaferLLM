"""Stage 4 gate: rope_body must compile to identical bytes on both PEs.

The transplant is only safe if every address baked into rope_body's instructions
means the same thing on the storage PE (which emits the bytes) and the compute PE
(which executes them). Rather than trying to enumerate which globals get embedded,
this compares the emitted bytes directly: if they match, every embedded reference
agrees, whatever those references turn out to be.

On mismatch it prints the differing 32-bit words and their delta. A delta equal to
the gap between two pinned sections tells you which symbol is still floating.

Usage:
    cs_python check_overlay.py <artifact_dir> [<artifact_dir_b>]

One directory  -> compare storage vs compute within that build (the Stage 4 gate).
Two directories -> compare rope_body across two builds, for the Stage 4b test of
                   whether its branches are PC-relative or absolute.
"""

import glob
import re
import subprocess
import sys
from pathlib import Path

PINNED = (".rope_ptr_src", ".rope_freqs_cos", ".rope_freqs_sin", ".rope_cos_val",
          ".rope_sin_val", ".rope_dsd_1", ".rope_dsd_2",
          ".rope_tmp_1", ".rope_tmp_2", ".rope_tmp_3", ".rope_tmp_4",
          ".rope_tmp_1_dsd", ".rope_tmp_2_dsd", ".rope_tmp_3_dsd",
          ".rope_tmp_4_dsd", ".cand_rope", ".slot_code")

# Immutable DSDs (X_tmp_N_dsd) are folded into immediates plus the backing
# array's base address, so they never become objects and their sections are
# never emitted. The pins stay declared in case a future config makes them
# real, but their absence is expected, not a failure.
OPTIONAL = (".rope_tmp_1_dsd", ".rope_tmp_2_dsd",
            ".rope_tmp_3_dsd", ".rope_tmp_4_dsd")


def sections(elf):
    out = subprocess.run(["readelf", "-SW", elf], capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m:
            res[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16))
    return res


def symbol(elf, name):
    out = subprocess.run(["readelf", "-sW", elf], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 8 and p[7] == name:
            return int(p[1], 16), int(p[2])
    return None


def func_bytes(elf, name):
    """Read a function's bytes out of the ELF image by address -> file offset."""
    hit = symbol(elf, name)
    if hit is None:
        return None, None, None
    addr, size = hit
    for name, (sh_addr, sh_off, sh_size) in sections(elf).items():
        # Debug sections are unallocated and sit at addr 0 with real sizes, so
        # they spuriously "contain" any low address. Only allocated sections
        # describe the PE's memory image.
        if sh_addr == 0 or name.startswith(".debug"):
            continue
        if sh_size and sh_addr <= addr < sh_addr + sh_size:
            off = sh_off + (addr - sh_addr)
            return Path(elf).read_bytes()[off:off + size], addr, size
    return None, addr, size


def classify(elfs):
    """Split ELFs into (storage, compute) by whether they define .cand_rope."""
    storage, compute = [], []
    for e in elfs:
        (storage if ".cand_rope" in sections(e) else compute).append(e)
    return storage, compute


def words(b):
    return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]


def compare(a_elf, b_elf, a_label, b_label):
    a, a_addr, a_size = func_bytes(a_elf, "rope_body")
    b, b_addr, b_size = func_bytes(b_elf, "rope_body")
    if a is None or b is None:
        print(f"  FAIL: rope_body not found in {a_label if a is None else b_label}")
        return False
    print(f"  {a_label:<9} rope_body @0x{a_addr:04x}  {a_size} B")
    print(f"  {b_label:<9} rope_body @0x{b_addr:04x}  {b_size} B")
    if a == b:
        print(f"  bytes IDENTICAL ({len(a)} B)")
        return True
    wa, wb = words(a), words(b)
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(wa, wb)) if x != y]
    print(f"  bytes DIFFER: {len(diffs)}/{len(wa)} words")
    for i, x, y in diffs[:20]:
        print(f"    word {i:>4} (+0x{i*4:04x}): {x:#010x} vs {y:#010x}   delta {y-x:+#x}")
    if len(diffs) > 20:
        print(f"    ... {len(diffs)-20} more")
    return False


def main():
    dirs = sys.argv[1:]
    elfs = sorted(glob.glob(f"{dirs[0]}/bin/out_*.elf"))
    storage, compute = classify(elfs)
    print(f"{dirs[0]}: {len(storage)} storage image(s), {len(compute)} compute image(s)")

    ok = True
    if len(dirs) == 1:
        print("\n[Stage 4 gate] storage vs compute, same build")
        if not storage:
            print("  FAIL: no ELF defines .cand_rope")
            return 1
        ok &= compare(storage[0], compute[0], "storage", "compute")

        print("\n[pinned section addresses]")
        s_sec, c_sec = sections(storage[0]), sections(compute[0])
        for name in PINNED:
            sa = s_sec.get(name, (None,))[0]
            ca = c_sec.get(name, (None,))[0]
            if sa is None and ca is None:
                note = "expected, folded to immediate" if name in OPTIONAL else "<-- pinned but never emitted"
                print(f"  {name:<18} absent from both  ({note})")
                ok = ok and name in OPTIONAL
            elif name in (".cand_rope", ".slot_code"):
                where = "storage" if sa is not None else "compute"
                print(f"  {name:<18} 0x{(sa or ca):04x}  ({where} only, expected)")
            elif sa != ca:
                print(f"  {name:<18} storage=0x{sa if sa else 0:04x} compute=0x{ca if ca else 0:04x}  MISMATCH")
                ok = False
            else:
                print(f"  {name:<18} 0x{sa:04x}  match")
    else:
        print("\n[Stage 4b] rope_body across two link addresses")
        s2 = classify(sorted(glob.glob(f"{dirs[1]}/bin/out_*.elf")))[0]
        ok &= compare(storage[0], s2[0], dirs[0], dirs[1])
        print("\n  identical -> no self-referencing address in the bytes;")
        print("               the loop's backward branch and the `if` forward branch")
        print("               are relocation-invariant (PC-relative).")
        print("  differ    -> absolute encoding; the offsets above localise it.")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
