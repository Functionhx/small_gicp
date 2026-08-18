#!/usr/bin/env python3
# Generate the README figures from real repository data:
#   hero.png              - real KITTI scan (raw) + two real scans fused by the GPU trajectory
#   benchmark.png         - CPU vs GPU end-to-end latency bars (numbers from BENCHMARK_GPU.md)
#   kitti00_trajectory.png- GT vs upstream CPU vs GPU trajectories over the full 4541-frame run
#
# Inputs (all real, produced by cuda/bench/odometry_gpu + scripts/eval_traj.py):
#   ~/datasets/kitti/odometry/velodyne/00/*.bin   official KITTI-00 velodyne frames
#   /tmp/full_cpu_gicp.txt                        upstream CPU chained trajectory
#   /tmp/full_full-gpu_gicp.txt                   GPU chained trajectory
#   /tmp/gt_kitti.txt                             official KITTI GT poses
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.expanduser("~/datasets/kitti/odometry/velodyne/00")


def read_bin(name):
    p = np.fromfile(os.path.join(DATA, name), dtype=np.float32).reshape(-1, 4)
    return p[:, :3]


def load_traj(path, n):
    ts = []
    for line in open(path):
        v = list(map(float, line.split()))
        if len(v) != 12:
            continue
        T = np.eye(4)
        T[:3, :4] = np.array(v).reshape(3, 4)
        ts.append(T)
        if len(ts) >= n:
            break
    return ts


# ---------------------------------------------------------------- hero.png
# Left: raw frame 000000. Right: frames 000000 + 000010 fused by the GPU-estimated
# relative pose (line 10 of the GPU trajectory) - two scans becoming one map.
def hero():
    rng = np.random.default_rng(0)
    a = read_bin("000000.bin")
    b = read_bin("000010.bin")
    T10 = load_traj("/tmp/full_full-gpu_gicp.txt", 11)[10]
    b_map = (T10[:3, :3] @ b.T).T + T10[:3, 3]

    def clip_near(pts):
        m = np.linalg.norm(pts, axis=1) < 45.0
        return pts[m]

    a, b_map = clip_near(a), clip_near(b_map)
    sa = a[rng.choice(len(a), min(30000, len(a)), replace=False)]
    sb = b_map[rng.choice(len(b_map), min(30000, len(b_map)), replace=False)]

    fig = plt.figure(figsize=(16, 7), dpi=100)
    fig.patch.set_facecolor("#0b0f1a")

    def style(ax, title):
        ax.set_facecolor("#0b0f1a")
        ax.set_axis_off()
        ax.view_init(elev=28, azim=-60)
        ax.set_box_aspect((2, 2, 0.9))
        ax.set_title(title, color="#e6edf3", fontsize=15, pad=2, fontweight="bold")

    ax1 = fig.add_subplot(121, projection="3d")
    style(ax1, "raw LiDAR scan")
    ax1.scatter(sa[:, 0], sa[:, 1], sa[:, 2], s=0.55, c=sa[:, 2], cmap="cool", linewidths=0, depthshade=False)

    ax2 = fig.add_subplot(122, projection="3d")
    style(ax2, "registered (2 scans, GPU pose)")
    ax2.scatter(sa[:, 0], sa[:, 1], sa[:, 2], s=0.55, c="#3fb6ff", linewidths=0, depthshade=False, alpha=0.55)
    ax2.scatter(sb[:, 0], sb[:, 1], sb[:, 2], s=0.55, c="#ff7a45", linewidths=0, depthshade=False, alpha=0.55)

    fig.text(0.5, 0.585, "CUDA", ha="center", color="#76b900", fontsize=30, fontweight="bold")
    fig.text(0.5, 0.415, "GICP / VGICP", ha="center", color="#e6edf3", fontsize=13)
    ax_arrow = fig.add_axes([0.40, 0.44, 0.20, 0.10])
    ax_arrow.set_axis_off()
    ax_arrow.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5), arrowprops=dict(arrowstyle="-|>", color="#76b900", lw=2.5))
    fig.suptitle("GPU-native point cloud registration", color="#e6edf3", fontsize=21, y=0.97, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "hero.png"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- benchmark.png
# End-to-end msec/frame: downsampling + covariance + full LM loop (BENCHMARK_GPU.md).
def benchmark():
    rows = [
        ("GICP · RTX 4070", 43.9, 6.0, "7.3x"),
        ("VGICP · RTX 4070", 34.4, 5.6, "6.1x"),
        ("GICP · Jetson Orin NX", 118.3, 16.5, "7.2x"),
        ("VGICP · Jetson Orin NX", 102.3, 14.2, "7.2x"),
    ]
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = np.arange(len(rows))[::-1]
    cpu = [r[1] for r in rows]
    gpu = [r[2] for r in rows]
    ax.barh(y + 0.19, cpu, height=0.36, color="#8b949e", label="upstream small_gicp (CPU)")
    ax.barh(y - 0.19, gpu, height=0.36, color="#76b900", label="small_gicp-cuda (GPU)")
    for yi, r in zip(y, rows):
        ax.text(r[1] + 1.5, yi + 0.19, f"{r[1]:.1f} ms", va="center", fontsize=12, color="#24292f")
        ax.text(r[2] + 1.5, yi - 0.19, f"{r[2]:.1f} ms", va="center", fontsize=12, color="#24292f", fontweight="bold")
        ax.text(131, yi, r[3], va="center", ha="left", fontsize=17, color="#1a7f37", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlim(0, 150)
    ax.set_xlabel("end-to-end latency per frame [msec]  (downsampling + covariance + full LM loop)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    ax.set_title("KITTI-00, 120k-point frames, 100-frame official protocol", fontsize=14, loc="left", color="#24292f")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark.png"), facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- kitti00_trajectory.png
def trajectory():
    gt = np.array([t[:3, 3] for t in load_traj("/tmp/gt_kitti.txt", 4541)])
    cpu = np.array([t[:3, 3] for t in load_traj("/tmp/full_cpu_gicp.txt", 4541)])
    gpu = np.array([t[:3, 3] for t in load_traj("/tmp/full_full-gpu_gicp.txt", 4541)])

    fig, ax = plt.subplots(figsize=(14, 6.6), dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(gt[:, 0], gt[:, 1], color="#8b949e", lw=2.2, label="ground truth")
    ax.plot(cpu[:, 0], cpu[:, 1], color="#1f77b4", lw=1.6, label="upstream CPU (GICP)")
    ax.plot(gpu[:, 0], gpu[:, 1], color="#d29922", lw=1.1, ls=(0, (3, 2)), label="small_gicp-cuda (GPU)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.set_title("KITTI-00 odometry: 4,541 frames, 3.7 km, frame-to-frame, no loop closure", fontsize=14, loc="left", color="#24292f")

    # Inset: a late-sequence corner where CPU and GPU stay glued while both drift off GT.
    axi = ax.inset_axes([0.58, 0.06, 0.36, 0.52])
    sl = slice(3500, 4300)
    axi.plot(gt[sl, 0], gt[sl, 1], color="#8b949e", lw=2.2)
    axi.plot(cpu[sl, 0], cpu[sl, 1], color="#1f77b4", lw=1.6)
    axi.plot(gpu[sl, 0], gpu[sl, 1], color="#d29922", lw=1.1, ls=(0, (3, 2)))
    axi.set_aspect("equal")
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_color("#d0d7de")
    axi.set_title("zoom: GPU (dashed) tracks CPU to 0.22%", fontsize=10, color="#57606a")
    ax.indicate_inset_zoom(axi, edgecolor="#d0d7de")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kitti00_trajectory.png"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    benchmark()
    trajectory()
    print("figures written to", os.path.abspath(OUT))
