#!/usr/bin/env python3
# Extract KITTI odometry velodyne sequence 00 frames directly from the official S3 zip
# via HTTP Range requests (no registration, no 80GB download).
import os
import struct
import sys
import urllib.request
import zlib

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
OUT = os.path.expanduser("~/datasets/kitti/odometry/velodyne/00")
MAX_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def ranged_get(start, length):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def total_size():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def parse_central_directory(size):
    # Fetch the tail and locate the (zip64) EOCD
    tail_len = 1 << 20
    tail = ranged_get(size - tail_len, tail_len)

    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD not found")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
    total_entries = struct.unpack("<H", tail[eocd + 10 : eocd + 12])[0]

    if cd_offset == 0xFFFFFFFF or total_entries == 0xFFFF:
        # zip64: locator sits just before EOCD
        loc = tail.rfind(b"PK\x06\x07")
        z64_eocd_offset = struct.unpack("<Q", tail[loc + 8 : loc + 16])[0]
        z64 = ranged_get(z64_eocd_offset, 56)
        assert z64[:4] == b"PK\x06\x06"
        total_entries = struct.unpack("<Q", z64[32:40])[0]
        cd_size = struct.unpack("<Q", z64[40:48])[0]
        cd_offset = struct.unpack("<Q", z64[48:56])[0]

    print(f"zip: {total_entries} entries, central directory at {cd_offset} ({cd_size} bytes)", file=sys.stderr)
    cd = ranged_get(cd_offset, cd_size)

    entries = {}
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
        comp_size, uncomp_size = struct.unpack("<II", cd[pos + 20 : pos + 28])
        name_len, extra_len, comment_len = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        lho = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        name = cd[pos + 46 : pos + 46 + name_len].decode()
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        # zip64 extra field may carry the real sizes/offset
        if lho == 0xFFFFFFFF or comp_size == 0xFFFFFFFF:
            e = 0
            while e + 4 <= len(extra):
                hid, hsize = struct.unpack("<HH", extra[e : e + 4])
                if hid == 1:
                    vals = []
                    off = e + 4
                    for want in (uncomp_size == 0xFFFFFFFF, comp_size == 0xFFFFFFFF, lho == 0xFFFFFFFF):
                        if want:
                            vals.append(struct.unpack("<Q", extra[off : off + 8])[0])
                            off += 8
                    vi = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = vals[vi]; vi += 1
                    if comp_size == 0xFFFFFFFF:
                        comp_size = vals[vi]; vi += 1
                    if lho == 0xFFFFFFFF:
                        lho = vals[vi]; vi += 1
                    break
                e += 4 + hsize
        entries[name] = (lho, comp_size, method)
        pos += 46 + name_len + extra_len + comment_len
    return entries


def extract(entry, out_path):
    lho, comp_size, method = entry
    # Local header: 30 bytes + name + extra; fetch a bit more to cover them
    head = ranged_get(lho, 30 + 512)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    data_start = lho + 30 + name_len + extra_len
    # Adjust: the local extra length can differ from the central one
    if data_start + comp_size > lho + len(head):
        data = head[data_start - lho :] + ranged_get(lho + len(head), comp_size - (len(head) - (data_start - lho)))
    else:
        data = head[data_start - lho : data_start - lho + comp_size]
    if method == 0:
        raw = data
    elif method == 8:
        raw = zlib.decompress(data, -15)
    else:
        raise RuntimeError(f"unsupported zip method {method}")
    with open(out_path, "wb") as f:
        f.write(raw)
    return len(raw)


def main():
    os.makedirs(OUT, exist_ok=True)
    size = total_size()
    print(f"remote zip size: {size / 1e9:.1f} GB", file=sys.stderr)
    entries = parse_central_directory(size)

    wanted = [n for n in entries if n.startswith("dataset/sequences/00/velodyne/") and n.endswith(".bin")]
    wanted.sort()
    print(f"sequence 00 velodyne frames available: {len(wanted)}, fetching first {MAX_FRAMES}", file=sys.stderr)

    for i, name in enumerate(wanted[:MAX_FRAMES]):
        out = os.path.join(OUT, os.path.basename(name))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        n = extract(entries[name], out)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{min(MAX_FRAMES, len(wanted))} ({n} bytes last)", file=sys.stderr, flush=True)
    print("done:", len(os.listdir(OUT)), "frames in", OUT, file=sys.stderr)


if __name__ == "__main__":
    main()
