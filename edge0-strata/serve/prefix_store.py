"""A content-addressed store of shared prompt prefixes (T7).

The engine can pin a shared prefix (`strata_prefix` -> `pin=N`) and save/restore its whole
running state as a session file (`POST /slots/0?action=save|restore`).  Both already work:
a pinned prefix was measured at 1.7 s against 21.9 s cold, and a restore of the same state
into a fresh process cost 180 ms of I/O for a 265 MB image.

What was missing is *finding* the image: the slot API takes a client-chosen file name, so
nothing ever looked one up.  This module keys an image by the prompt's own tokens, so the
second request that shares a prefix finds it without the client naming anything.

Design notes:

* The key is a hash of the model name and the token ids of the prefix - not of the text, so
  two clients that spell the same prompt differently still share an entry, and a tokenizer's
  merge across a message boundary cannot alias two different prefixes.
* Sizes are bounded by BYTES, not by count.  An image is ~139 KiB per token of the whole session
  it was saved from (265 MB for 1,899 tokens on the 120B pack), so a count alone is not a disk
  budget: eight 265 MB images and eight 4 MB images both pass `max_files=8` while differing 60x
  in bytes.  `max_total_bytes` is the real limit and `max_files` stays as a cheap secondary bound.
* Every failure is non-fatal by design: a store that cannot read or write must degrade to the
  engine's own prompt cache, never fail a request.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time

KEY_CHARS = 40          # 160 bits of sha256: collision-free for any plausible number of prompts

STORE_SHARE = 4         # an unconfigured store may take a quarter of the free space on its volume
STORE_MAX_BYTES = 8 << 30    # ... and never more than this: past it the images are a cache, not a data set
TYPICAL_IMAGE_BYTES = 256 << 20   # one 1,899-token session image on the 120B pack is 265 MB


def free_bytes(path: str) -> int:
    """Free space on the volume holding `path` (or its nearest existing parent).  0 when unknown."""
    p = os.path.abspath(path)
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return 0


def default_total_bytes(path: str) -> int:
    """The byte budget an unconfigured store gets on `path`: `STORE_SHARE` of the free space, capped at
    `STORE_MAX_BYTES`.  0 (no cap) only when the free space cannot be read.

    A fixed default cannot be right everywhere: 8 GiB is most of a laptop's free space and 3% of a
    4 TB data drive.  Deriving it keeps the store proportional to the disk it actually lives on - a
    cache that insists on its own number is how a full disk happens.
    """
    free = free_bytes(path)
    if not free:
        return 0
    return min(free // STORE_SHARE, STORE_MAX_BYTES)


def default_max_files(total_bytes: int, max_files: int = 8) -> int:
    """How many images a `total_bytes` budget is worth, assuming a `TYPICAL_IMAGE_BYTES` image: at least one,
    at most `max_files`.  0 (unknown) keeps the count at `max_files`."""
    if total_bytes <= 0:
        return max_files
    return max(1, min(max_files, total_bytes // TYPICAL_IMAGE_BYTES))


def prefix_key(model: str, ids, n: int) -> str:
    """The store key for the first `n` token ids of a prompt under `model`.

    Ids are hashed as their little-endian 4-byte forms rather than their decimal text: a
    text join would let [1, 23] and [12, 3] hash alike.
    """
    if n <= 0 or n > len(ids):
        raise ValueError(f"prefix_key: {n} of {len(ids)} ids")
    h = hashlib.sha256()
    h.update(model.encode("utf-8", "surrogatepass"))
    h.update(b"\x00")
    h.update(memoryview(bytearray().join(int(i).to_bytes(4, "little", signed=True) for i in ids[:n])))
    return h.hexdigest()[:KEY_CHARS]


class PrefixStore:
    """A directory of `<key>.sess` images, bounded by count and evicted least-recently-used."""

    SUFFIX = ".sess"

    def __init__(self, directory: str, max_files: int = 8, max_image_bytes: int = 0,
                 max_total_bytes: int = 0, auto_total: bool = False):
        self.directory = directory
        self.max_files = max(1, int(max_files))
        # 0 = no per-image cap.  The engine writes its whole live session (~139 KiB per prompt
        # token on the 120B pack), not just the pinned prefix, so an unbounded store is the real
        # risk: a cap keeps one oversized prefix from filling the directory.
        self.max_image_bytes = max(0, int(max_image_bytes))
        # 0 = no total cap.  This is the bound that actually protects the disk: a session image
        # grows with the prompt, so counting files says nothing about how much room they take.
        self.max_total_bytes = max(0, int(max_total_bytes))
        # `auto_total`: the budget is not a setting but a reading of the disk, so it is taken again
        # whenever the store grows (`refresh_total`).  A fixed number picked at startup is wrong as
        # soon as anything else on the volume fills or empties: it either hoards room the machine
        # needs or refuses room the machine has.
        self.auto_total = bool(auto_total)

    def refresh_total(self) -> int:
        """Re-read the volume and update the budget when it is derived from free space.

        Called before the store grows (after a save, before eviction): a shrinking volume evicts the
        older images there and then, and a volume that has freed room lets the store keep more.  A
        fixed `max_total_bytes` is left alone.  Returns the budget now in force.
        """
        if self.auto_total:
            self.max_total_bytes = default_total_bytes(self.directory)
        return self.max_total_bytes

    def check_image(self, key: str) -> tuple[bool, int]:
        """(fits the cap, size in bytes) for a saved entry.  Missing/unreadable counts as 0."""
        try:
            size = os.stat(self.path(key)).st_size
        except OSError:
            return True, 0
        return (self.max_image_bytes == 0 or size <= self.max_image_bytes), size

    def drop(self, key: str) -> bool:
        try:
            os.remove(self.path(key))
            return True
        except OSError:
            return False

    def path(self, key: str) -> str:
        return os.path.join(self.directory, key + self.SUFFIX)

    def has(self, key: str) -> bool:
        try:
            st = os.lstat(self.path(key))
        except OSError:
            return False
        return st.st_size > 0 and not os.path.islink(self.path(key))

    def touch(self, key: str) -> None:
        """Mark the entry as just used (the LRU order is mtime)."""
        try:
            os.utime(self.path(key), None)
        except OSError:
            pass

    def entries(self) -> list[tuple[float, str]]:
        out = []
        try:
            with os.scandir(self.directory) as it:
                for e in it:
                    if not e.name.endswith(self.SUFFIX) or not e.is_file(follow_symlinks=False):
                        continue
                    try:
                        out.append((e.stat(follow_symlinks=False).st_mtime, e.path))
                    except OSError:
                        continue
        except OSError:
            return []
        out.sort()
        return out

    def evict(self, keep: str | None = None) -> int:
        """Drop the oldest images until the store fits BOTH bounds - at most `max_files` entries and
        at most `max_total_bytes` on disk - never dropping `keep`.

        Returns how many were removed.  Called after a save, when the new entry is the newest and
        therefore never the one dropped.  The byte bound is checked after every removal because the
        oldest image can be most of the store: one 265 MB entry leaving a 300 MB budget fits, the
        next one may not.
        """
        entries = self.entries()
        sizes: dict[str, int] = {}
        total = 0
        for _mtime, p in entries:
            try:
                sizes[p] = os.stat(p).st_size
            except OSError:
                sizes[p] = 0
            total += sizes[p]
        removed = 0
        while entries and (len(entries) > self.max_files or
                           (self.max_total_bytes > 0 and total > self.max_total_bytes)):
            _mtime, p = entries.pop(0)
            if keep is not None and p == self.path(keep):
                entries.append((_mtime, p))
                break
            try:
                os.remove(p)
                removed += 1
                total -= sizes.get(p, 0)
            except OSError:
                entries.append((_mtime, p))
                break
        return removed

    def note(self) -> str:
        """One line for the log: how many images the store holds and how large they are."""
        entries = self.entries()
        total = 0
        for _m, p in entries:
            try:
                total += os.stat(p).st_size
            except OSError:
                pass
        return f"{len(entries)} image(s), {total / (1 << 20):.0f} MiB in {self.directory}"
