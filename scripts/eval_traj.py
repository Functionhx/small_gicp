#!/usr/bin/env python3
# Self-contained KITTI-format trajectory evaluation: APE and RPE(100/400/800).
# Metric definitions follow the KITTI devkit convention used in small_gicp's BENCHMARK.md.
# Usage: eval_traj.py <traj_a.txt> <traj_b.txt> [--rpe 100,400,800]

import argparse
import math
import sys

import numpy as np


def load_traj(path):
    poses = []
    with open(path) as f:
        for line in f:
            v = [float(x) for x in line.split()]
            if len(v) != 12:
                continue
            T = np.eye(4)
            T[:3, :4] = np.array(v).reshape(3, 4)
            poses.append(T)
    return poses


def frame_error(T_a, T_b):
    """Translation [m] and rotation [deg] of inv(T_a) @ T_b."""
    d = np.linalg.inv(T_a) @ T_b
    t = np.linalg.norm(d[:3, 3])
    c = np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    r = math.degrees(math.acos(c))
    return t, r


def ape(a, b):
    errs = [frame_error(x, y) for x, y in zip(a, b)]
    return np.array([e[0] for e in errs]), np.array([e[1] for e in errs])


def rpe(a, b, step):
    t_errs, r_errs = [], []
    for i in range(len(a) - step):
        da = np.linalg.inv(a[i]) @ a[i + step]
        db = np.linalg.inv(b[i]) @ b[i + step]
        d = np.linalg.inv(da) @ db
        t_errs.append(np.linalg.norm(d[:3, 3]))
        c = np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        r_errs.append(math.degrees(math.acos(c)))
    return np.array(t_errs), np.array(r_errs)


def stats(x):
    return f"{x.mean():.3f} +- {x.std():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_a")
    ap.add_argument("traj_b")
    ap.add_argument("--rpe", default="100,400,800")
    args = ap.parse_args()

    a = load_traj(args.traj_a)
    b = load_traj(args.traj_b)
    n = min(len(a), len(b))
    if n == 0:
        print("empty trajectory", file=sys.stderr)
        return 1
    a, b = a[:n], b[:n]

    t, r = ape(a, b)
    print(f"pairs={n}")
    print(f"APE_trans[m]: {stats(t)}  max={t.max():.3f}")
    print(f"APE_rot[deg]: {stats(r)}  max={r.max():.3f}")
    for step in [int(s) for s in args.rpe.split(",")]:
        if n > step:
            rt, rr = rpe(a, b, step)
            print(f"RPE({step})_trans[m]: {stats(rt)}")
            print(f"RPE({step})_rot[deg/km]: {stats(rr / max(1e-9, step * 0.1))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
