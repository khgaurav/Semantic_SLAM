#!/usr/bin/env python3
"""Visualize the hybrid semantic map built by the mapping pipeline.

Generates a multi-panel figure with:
  - 3D trajectory with color-coded keyframes
  - 2D bird's-eye view (X-Y)
  - Statistics annotation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import faiss
import os


def main():
    map_dir = "/home/gauravkh/ros2_ws/data/hybrid_map"
    poses_path = os.path.join(map_dir, "keyframe_poses.npy")
    ids_path = os.path.join(map_dir, "keyframe_ids.npy")
    index_path = os.path.join(map_dir, "map_index.faiss")

    if not os.path.exists(poses_path):
        print(f"Map poses not found at {poses_path}")
        return

    poses = np.load(poses_path)
    if len(poses) == 0:
        print("Poses array is empty!")
        return

    ids = np.load(ids_path) if os.path.exists(ids_path) else np.arange(len(poses))
    index = faiss.read_index(index_path) if os.path.exists(index_path) else None

    x, y, z = poses[:, 0], poses[:, 1], poses[:, 2]

    # Compute statistics
    diffs = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    total_dist = np.sum(diffs)
    n_kf = len(poses)
    emb_dim = index.d if index else "N/A"
    faiss_n = index.ntotal if index else "N/A"

    # Color by keyframe index
    colors = cm.viridis(np.linspace(0, 1, n_kf))

    fig = plt.figure(figsize=(16, 7), facecolor='#1a1a2e')
    fig.suptitle("Hybrid Semantic Map — Keyframe Trajectory", fontsize=16,
                 fontweight='bold', color='white', y=0.98)

    # --- 3D trajectory ---
    ax1 = fig.add_subplot(121, projection='3d', facecolor='#1a1a2e')
    ax1.plot(x, y, z, color='#555577', linewidth=0.5, alpha=0.5, zorder=1)
    ax1.scatter(x, y, z, c=np.arange(n_kf), cmap='viridis', s=12, alpha=0.9, zorder=2)
    ax1.scatter(x[0], y[0], z[0], color='lime', marker='o', s=80, label='Start', edgecolors='white', zorder=3)
    ax1.scatter(x[-1], y[-1], z[-1], color='red', marker='X', s=80, label='End', edgecolors='white', zorder=3)

    # Equal aspect ratio
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2, (z.max() + z.min()) / 2
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    ax1.set_xlabel('X [m]', color='white', fontsize=9)
    ax1.set_ylabel('Y [m]', color='white', fontsize=9)
    ax1.set_zlabel('Z [m]', color='white', fontsize=9)
    ax1.set_title('3D Trajectory', color='white', fontsize=12, pad=10)
    ax1.legend(fontsize=8, loc='upper right', facecolor='#2a2a4e', edgecolor='white',
               labelcolor='white')
    ax1.tick_params(colors='#aaaaaa', labelsize=7)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # --- 2D bird's-eye view ---
    ax2 = fig.add_subplot(122, facecolor='#16213e')
    sc = ax2.scatter(x, y, c=np.arange(n_kf), cmap='viridis', s=15, alpha=0.9)
    ax2.plot(x, y, color='#555577', linewidth=0.5, alpha=0.5, zorder=0)
    ax2.scatter(x[0], y[0], color='lime', marker='o', s=80, label='Start', edgecolors='white', zorder=3)
    ax2.scatter(x[-1], y[-1], color='red', marker='X', s=80, label='End', edgecolors='white', zorder=3)
    ax2.set_xlabel('X [m]', color='white', fontsize=10)
    ax2.set_ylabel('Y [m]', color='white', fontsize=10)
    ax2.set_title("Bird's Eye View (X-Y)", color='white', fontsize=12)
    ax2.legend(fontsize=8, loc='upper right', facecolor='#2a2a4e', edgecolor='white',
               labelcolor='white')
    ax2.tick_params(colors='#aaaaaa', labelsize=8)
    ax2.set_aspect('equal')
    cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label('Keyframe Index', color='white', fontsize=9)
    cbar.ax.tick_params(colors='#aaaaaa', labelsize=7)

    # Stats text box
    stats_text = (
        f"Keyframes: {n_kf}\n"
        f"Embedding dim: {emb_dim}\n"
        f"FAISS vectors: {faiss_n}\n"
        f"Total path: {total_dist:.1f} m\n"
        f"X range: [{x.min():.1f}, {x.max():.1f}]\n"
        f"Y range: [{y.min():.1f}, {y.max():.1f}]\n"
        f"Z range: [{z.min():.1f}, {z.max():.1f}]"
    )
    fig.text(0.50, 0.02, stats_text, fontsize=8, color='#cccccc',
             family='monospace', ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.9, edgecolor='#444466'))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    output_path = "/home/gauravkh/ros2_ws/data/hybrid_map_trajectory.png"
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Saved map visualization to {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
