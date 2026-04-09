#!/usr/bin/env python3
"""
Comprehensive verification and visualization of the hybrid localization system.

Produces:
  1. Trajectory plot (3D + bird's eye) with good/bad keyframe annotations
  2. Embedding similarity matrix heatmap
  3. t-SNE of embeddings colored by trajectory position
  4. Sample images extracted from bag with their matched keyframes
  5. Localization accuracy test: query images vs. retrieved keyframe poses
"""

import numpy as np
import faiss
import os
import sys
import sqlite3
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────────
MAP_DIR = "/home/gauravkh/ros2_ws/data/hybrid_map"
BAG_DB  = "/home/gauravkh/ros2_ws/data/gate_01_mcap/gate_01_mcap.db3"
OUT_DIR = "/home/gauravkh/ros2_ws/data/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 0 — Load map data
# ══════════════════════════════════════════════════════════════════════════════
def load_map():
    poses = np.load(os.path.join(MAP_DIR, "keyframe_poses.npy"))
    ids   = np.load(os.path.join(MAP_DIR, "keyframe_ids.npy"))
    index = faiss.read_index(os.path.join(MAP_DIR, "map_index.faiss"))
    print(f"Loaded map: {index.ntotal} keyframes, dim={index.d}, poses={poses.shape}")
    return poses, ids, index


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — Trajectory quality analysis & plot
# ══════════════════════════════════════════════════════════════════════════════
def analyze_trajectory(poses, ids):
    n = len(poses)
    diffs = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)

    # Mark "good" keyframes: no step > 20m from previous
    good_mask = np.ones(n, dtype=bool)
    for i in range(1, n):
        if diffs[i - 1] > 20.0:
            good_mask[i:] = False
            break
    # Also mark origin-stuck poses
    origin_mask = np.all(np.abs(poses[:, :3]) < 1e-4, axis=1)
    good_mask &= ~origin_mask

    n_good = good_mask.sum()
    n_bad  = n - n_good
    print(f"Trajectory: {n_good} good / {n_bad} bad keyframes")
    if n_good > 1:
        good_diffs = np.linalg.norm(np.diff(poses[good_mask][:, :3], axis=0), axis=1)
        print(f"  Good path length: {good_diffs.sum():.1f} m")
        print(f"  Good X range: [{poses[good_mask, 0].min():.1f}, {poses[good_mask, 0].max():.1f}]")
        print(f"  Good Y range: [{poses[good_mask, 1].min():.1f}, {poses[good_mask, 1].max():.1f}]")

    # ── Plot ──
    fig = plt.figure(figsize=(18, 8), facecolor='#1a1a2e')
    fig.suptitle("Hybrid Semantic Map — Trajectory Quality Analysis",
                 fontsize=15, fontweight='bold', color='white', y=0.98)

    # 3D (good only)
    ax1 = fig.add_subplot(131, projection='3d', facecolor='#1a1a2e')
    gp = poses[good_mask]
    if len(gp) > 1:
        c = np.linspace(0, 1, len(gp))
        ax1.plot(gp[:, 0], gp[:, 1], gp[:, 2], color='#555577', lw=0.5, alpha=0.5)
        ax1.scatter(gp[:, 0], gp[:, 1], gp[:, 2], c=c, cmap='viridis', s=10, alpha=0.9)
        ax1.scatter(*gp[0, :3], color='lime', s=80, marker='o', label='Start', edgecolors='w')
        ax1.scatter(*gp[-1, :3], color='red', s=80, marker='X', label='End', edgecolors='w')
    ax1.set_title('3D Trajectory (good keyframes)', color='white', fontsize=10)
    for ax_obj in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
        ax_obj.pane.fill = False
    ax1.tick_params(colors='#aaa', labelsize=7)
    ax1.set_xlabel('X [m]', color='white', fontsize=8)
    ax1.set_ylabel('Y [m]', color='white', fontsize=8)
    ax1.set_zlabel('Z [m]', color='white', fontsize=8)
    ax1.legend(fontsize=7, facecolor='#2a2a4e', edgecolor='white', labelcolor='white')

    # 2D bird's eye (good only)
    ax2 = fig.add_subplot(132, facecolor='#16213e')
    if len(gp) > 1:
        sc = ax2.scatter(gp[:, 0], gp[:, 1], c=np.arange(len(gp)), cmap='viridis', s=12, alpha=0.9)
        ax2.plot(gp[:, 0], gp[:, 1], color='#555577', lw=0.5, alpha=0.5)
        ax2.scatter(*gp[0, :2], color='lime', s=80, marker='o', edgecolors='w', zorder=5)
        ax2.scatter(*gp[-1, :2], color='red', s=80, marker='X', edgecolors='w', zorder=5)
        cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
        cbar.set_label('Keyframe Index', color='white', fontsize=8)
        cbar.ax.tick_params(colors='#aaa', labelsize=7)
    ax2.set_title("Bird's Eye (good keyframes)", color='white', fontsize=10)
    ax2.set_xlabel('X [m]', color='white', fontsize=8)
    ax2.set_ylabel('Y [m]', color='white', fontsize=8)
    ax2.set_aspect('equal')
    ax2.tick_params(colors='#aaa', labelsize=7)

    # Step-size histogram
    ax3 = fig.add_subplot(133, facecolor='#16213e')
    ax3.hist(diffs, bins=50, color='#4a90d9', edgecolor='#222', alpha=0.8, log=True)
    ax3.axvline(20, color='red', ls='--', lw=1.5, label='Divergence threshold (20m)')
    ax3.set_title('Step Size Distribution', color='white', fontsize=10)
    ax3.set_xlabel('Step size [m]', color='white', fontsize=8)
    ax3.set_ylabel('Count (log)', color='white', fontsize=8)
    ax3.legend(fontsize=7, facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    ax3.tick_params(colors='#aaa', labelsize=7)

    # Stats box
    stats = (f"Total keyframes: {n}\n"
             f"Good: {n_good}  Bad: {n_bad}\n"
             f"Path (good): {good_diffs.sum():.1f} m" if n_good > 1 else f"Total: {n}\nGood: {n_good}")
    fig.text(0.5, 0.01, stats, fontsize=8, color='#ccc', family='monospace',
             ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.9, edgecolor='#444'))

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    out = os.path.join(OUT_DIR, "01_trajectory_analysis.png")
    plt.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return good_mask


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — Embedding analysis (similarity matrix + t-SNE)
# ══════════════════════════════════════════════════════════════════════════════
def analyze_embeddings(index, poses, good_mask):
    n = index.ntotal
    dim = index.d

    # Reconstruct all embeddings from FAISS
    embeddings = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        embeddings[i] = faiss.rev_swig_ptr(index.get_xb(), n * dim).reshape(n, dim)[i]
    # Simpler: direct memory access
    embeddings = faiss.rev_swig_ptr(index.get_xb(), n * dim).reshape(n, dim).copy()

    # Cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a2e')
    fig.suptitle("Embedding Analysis", fontsize=15, fontweight='bold', color='white', y=0.98)

    # Similarity matrix
    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap='inferno', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_title('Pairwise Cosine Similarity', color='white', fontsize=11)
    ax.set_xlabel('Keyframe Index', color='white', fontsize=9)
    ax.set_ylabel('Keyframe Index', color='white', fontsize=9)
    ax.tick_params(colors='#aaa', labelsize=7)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Cosine Similarity', color='white', fontsize=8)
    cbar.ax.tick_params(colors='#aaa', labelsize=7)

    # Annotate good/bad boundary
    n_good = good_mask.sum()
    if n_good < n:
        first_bad = np.argmin(good_mask)
        ax.axhline(first_bad, color='red', ls='--', lw=0.8, alpha=0.7)
        ax.axvline(first_bad, color='red', ls='--', lw=0.8, alpha=0.7)
        ax.text(first_bad + 2, 5, 'SLAM diverged', color='red', fontsize=7)

    # t-SNE
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(30, n - 1), random_state=42)
        emb_2d = tsne.fit_transform(embeddings)
        sc = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1],
                         c=np.arange(n), cmap='viridis', s=15, alpha=0.8)
        # Highlight bad poses
        if n_good < n:
            ax2.scatter(emb_2d[~good_mask, 0], emb_2d[~good_mask, 1],
                        facecolors='none', edgecolors='red', s=30, lw=0.5, alpha=0.6,
                        label='Diverged poses')
            ax2.legend(fontsize=7, facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
        cbar2 = plt.colorbar(sc, ax=ax2, pad=0.02)
        cbar2.set_label('Keyframe Index', color='white', fontsize=8)
        cbar2.ax.tick_params(colors='#aaa', labelsize=7)
        ax2.set_title('t-SNE of SigLIP Embeddings', color='white', fontsize=11)
    except ImportError:
        ax2.text(0.5, 0.5, 'sklearn not installed\n(pip install scikit-learn)',
                 transform=ax2.transAxes, ha='center', color='white', fontsize=12)
        ax2.set_title('t-SNE (unavailable)', color='white', fontsize=11)
    ax2.tick_params(colors='#aaa', labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "02_embedding_analysis.png")
    plt.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # Print similarity stats
    diag_mask = ~np.eye(n, dtype=bool)
    print(f"Similarity stats (off-diagonal):")
    print(f"  Mean: {sim_matrix[diag_mask].mean():.4f}")
    print(f"  Min:  {sim_matrix[diag_mask].min():.4f}")
    print(f"  Max:  {sim_matrix[diag_mask].max():.4f}")
    print(f"  Std:  {sim_matrix[diag_mask].std():.4f}")

    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — Extract sample images from bag & test localization
# ══════════════════════════════════════════════════════════════════════════════
def extract_bag_images(bag_path, topic='/camera/color/image_raw/compressed',
                       max_images=20, stride=None):
    """Extract sample compressed images from a ROS2 bag (sqlite3 storage)."""
    import cv2

    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()

    # Get topic ID
    cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
    row = cursor.fetchone()
    if row is None:
        print(f"Topic {topic} not found in bag!")
        conn.close()
        return [], []
    topic_id = row[0]

    # Count messages
    cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (topic_id,))
    total = cursor.fetchone()[0]
    print(f"Found {total} images in bag for topic {topic}")

    if stride is None:
        stride = max(1, total // max_images)

    # Extract images
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                   (topic_id,))

    images = []
    timestamps = []
    for i, (ts, data) in enumerate(cursor):
        if i % stride != 0:
            continue
        if len(images) >= max_images:
            break
        try:
            # ROS2 CDR-serialized CompressedImage — we need to parse it
            # The CDR format has header + format string + data bytes
            # Simpler: use cv2 to decode the JPEG/PNG from the raw bytes
            # The compressed image data starts after the CDR header
            # Try to find JPEG header (FF D8) in the data
            data_bytes = bytes(data)
            jpeg_start = data_bytes.find(b'\xff\xd8')
            if jpeg_start == -1:
                # Try PNG header
                jpeg_start = data_bytes.find(b'\x89PNG')
            if jpeg_start == -1:
                continue

            img_bytes = data_bytes[jpeg_start:]
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                timestamps.append(ts)
        except Exception as e:
            print(f"  Failed to decode image {i}: {e}")
            continue

    conn.close()
    print(f"Extracted {len(images)} sample images")
    return images, timestamps


def test_localization(images, index, poses, ids):
    """Run localization on extracted images and return results."""
    import torch
    from PIL import Image as PilImage
    from transformers import AutoProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP 2 for localization test (device={device})...")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()

    results = []
    for i, img in enumerate(images):
        pil_img = PilImage.fromarray(img)
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = model.vision_model(**inputs)
            emb = vision_out.pooler_output
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        emb_np = emb.cpu().numpy().astype(np.float32)
        distances, indices = index.search(emb_np, 5)  # top-5

        best_idx = indices[0][0]
        best_score = distances[0][0]
        best_pose = poses[best_idx]
        best_kf_id = ids[best_idx]

        results.append({
            'query_idx': i,
            'best_kf_id': int(best_kf_id),
            'best_score': float(best_score),
            'best_pose': best_pose,
            'top5_scores': distances[0].tolist(),
            'top5_indices': indices[0].tolist(),
        })
        print(f"  Image {i:2d}: matched KF {best_kf_id:3d} (score={best_score:.4f}) "
              f"at ({best_pose[0]:.2f}, {best_pose[1]:.2f}, {best_pose[2]:.2f})")

    return results


def visualize_localization(images, results, poses, good_mask):
    """Create a grid of query images with their localization results."""
    n = min(len(images), 12)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), facecolor='#1a1a2e')
    fig.suptitle("Localization Test — Query Images & Matched Keyframes",
                 fontsize=14, fontweight='bold', color='white', y=0.99)

    for i in range(rows * cols):
        ax = axes.flat[i] if rows > 1 else (axes[i] if cols > 1 else axes)
        ax.set_facecolor('#16213e')
        if i >= n:
            ax.axis('off')
            continue

        ax.imshow(images[i])
        r = results[i]
        kf_id = r['best_kf_id']
        score = r['best_score']
        p = r['best_pose']
        is_good = good_mask[r['top5_indices'][0]] if r['top5_indices'][0] < len(good_mask) else False

        color = '#00ff00' if is_good else '#ff4444'
        ax.set_title(f"KF {kf_id} | score={score:.3f}\n"
                     f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})",
                     color=color, fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "03_localization_test.png")
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def visualize_localization_on_map(results, poses, good_mask):
    """Plot matched keyframe positions on the trajectory map."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    gp = poses[good_mask]
    bp = poses[~good_mask]

    # Plot full trajectory (good)
    if len(gp) > 1:
        ax.plot(gp[:, 0], gp[:, 1], color='#555577', lw=1, alpha=0.5, zorder=1)
        ax.scatter(gp[:, 0], gp[:, 1], c=np.arange(len(gp)), cmap='viridis',
                   s=8, alpha=0.6, zorder=2)

    # Plot matched keyframes
    matched_poses = np.array([r['best_pose'] for r in results])
    matched_scores = np.array([r['best_score'] for r in results])
    sc = ax.scatter(matched_poses[:, 0], matched_poses[:, 1],
                    c=matched_scores, cmap='RdYlGn', s=80, edgecolors='white',
                    lw=1.5, zorder=5, vmin=0.5, vmax=1.0)

    for i, r in enumerate(results):
        ax.annotate(f"Q{i}", (r['best_pose'][0], r['best_pose'][1]),
                    fontsize=6, color='white', ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Match Score', color='white', fontsize=9)
    cbar.ax.tick_params(colors='#aaa', labelsize=7)

    ax.set_title("Localization Results on Map", color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('X [m]', color='white', fontsize=10)
    ax.set_ylabel('Y [m]', color='white', fontsize=10)
    ax.set_aspect('equal')
    ax.tick_params(colors='#aaa', labelsize=8)

    # Stats
    stats = (f"Queries: {len(results)}\n"
             f"Mean score: {matched_scores.mean():.4f}\n"
             f"Min score: {matched_scores.min():.4f}\n"
             f"Max score: {matched_scores.max():.4f}")
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8,
            color='#ccc', family='monospace', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.9, edgecolor='#444'))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "04_localization_on_map.png")
    plt.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def print_summary(poses, good_mask, index, results):
    """Print a final summary report."""
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    n = len(poses)
    n_good = good_mask.sum()
    print(f"\n  Map Statistics:")
    print(f"    Total keyframes:    {n}")
    print(f"    Good keyframes:     {n_good} ({100*n_good/n:.0f}%)")
    print(f"    Bad keyframes:      {n - n_good} ({100*(n-n_good)/n:.0f}%)")
    print(f"    Embedding dim:      {index.d}")
    print(f"    FAISS index size:   {os.path.getsize(os.path.join(MAP_DIR, 'map_index.faiss'))/1024:.1f} KB")

    if n_good > 1:
        gp = poses[good_mask]
        diffs = np.linalg.norm(np.diff(gp[:, :3], axis=0), axis=1)
        print(f"    Good path length:   {diffs.sum():.1f} m")

    scores = [r['best_score'] for r in results]
    print(f"\n  Localization Test ({len(results)} queries):")
    print(f"    Mean match score:   {np.mean(scores):.4f}")
    print(f"    Min match score:    {np.min(scores):.4f}")
    print(f"    Max match score:    {np.max(scores):.4f}")

    # Check how many matched to good vs bad keyframes
    good_matches = sum(1 for r in results if good_mask[r['top5_indices'][0]])
    print(f"    Matched to good KF: {good_matches}/{len(results)}")
    print(f"    Matched to bad KF:  {len(results) - good_matches}/{len(results)}")

    print(f"\n  Issues Found:")
    if n - n_good > 0:
        print(f"    [CRITICAL] LIO-SAM trajectory diverged after keyframe ~{np.argmin(good_mask)}")
        print(f"               {n - n_good} keyframes have unrealistic positions")
    if n_good < 20:
        print(f"    [WARNING]  Only {n_good} good keyframes — map may be too sparse for reliable localization")

    # Check for duplicate poses
    unique_poses = len(np.unique(poses[:, :3].round(3), axis=0))
    if unique_poses < n:
        print(f"    [WARNING]  {n - unique_poses} duplicate poses in map (no keyframe filtering)")

    print(f"\n  Output visualizations saved to: {OUT_DIR}/")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Hybrid Localization — Verification & Visualization")
    print("=" * 70)

    # Load map
    poses, ids, index = load_map()

    # Part 1: Trajectory analysis
    print("\n--- Part 1: Trajectory Analysis ---")
    good_mask = analyze_trajectory(poses, ids)

    # Part 2: Embedding analysis
    print("\n--- Part 2: Embedding Analysis ---")
    analyze_embeddings(index, poses, good_mask)

    # Part 3: Extract images from bag & test localization
    print("\n--- Part 3: Image Extraction from Bag ---")
    if not os.path.exists(BAG_DB):
        print(f"Bag not found at {BAG_DB}, skipping localization test")
        print_summary(poses, good_mask, index, [])
        return

    images, timestamps = extract_bag_images(BAG_DB, max_images=20)
    if len(images) == 0:
        print("No images extracted, skipping localization test")
        print_summary(poses, good_mask, index, [])
        return

    print("\n--- Part 4: Localization Test ---")
    results = test_localization(images, index, poses, ids)

    print("\n--- Part 5: Visualization ---")
    visualize_localization(images, results, poses, good_mask)
    visualize_localization_on_map(results, poses, good_mask)

    print_summary(poses, good_mask, index, results)


if __name__ == '__main__':
    main()
