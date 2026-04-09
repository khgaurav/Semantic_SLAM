#!/usr/bin/env python3
"""
Clean the existing hybrid map by removing diverged poses, rebuild the FAISS
index, re-test localization, and produce final visualizations.
"""

import numpy as np
import faiss
import os
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import torch
from PIL import Image as PilImage
from transformers import AutoProcessor, AutoModel
import cv2

MAP_DIR = "/home/gauravkh/ros2_ws/data/hybrid_map"
CLEAN_DIR = "/home/gauravkh/ros2_ws/data/hybrid_map_clean"
BAG_DB = "/home/gauravkh/ros2_ws/data/gate_01_mcap/gate_01_mcap.db3"
OUT_DIR = "/home/gauravkh/ros2_ws/data/visualizations"
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_clean_map():
    """Load map, identify good keyframes, build cleaned FAISS index."""
    poses = np.load(os.path.join(MAP_DIR, "keyframe_poses.npy"))
    ids = np.load(os.path.join(MAP_DIR, "keyframe_ids.npy"))
    index = faiss.read_index(os.path.join(MAP_DIR, "map_index.faiss"))

    n = index.ntotal
    dim = index.d
    embeddings = faiss.rev_swig_ptr(index.get_xb(), n * dim).reshape(n, dim).copy()

    print(f"Original map: {n} keyframes, dim={dim}")

    # Identify good keyframes: step distance < 20m, not stuck at origin
    good_mask = np.ones(n, dtype=bool)

    # Remove origin-stuck poses
    origin = np.all(np.abs(poses[:, :3]) < 1e-4, axis=1)
    good_mask &= ~origin
    print(f"  Origin-stuck: {origin.sum()}")

    # Remove poses after first large jump
    diffs = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    for i in range(len(diffs)):
        if diffs[i] > 20.0:
            good_mask[i + 1:] = False
            print(f"  Divergence at keyframe {i+1} (jump={diffs[i]:.1f}m)")
            break

    # Remove duplicate poses (within 0.5m of previous good keyframe)
    good_indices = np.where(good_mask)[0]
    filtered_mask = np.zeros(n, dtype=bool)
    if len(good_indices) > 0:
        filtered_mask[good_indices[0]] = True
        last_good = good_indices[0]
        for idx in good_indices[1:]:
            dist = np.linalg.norm(poses[idx, :3] - poses[last_good, :3])
            if dist >= 0.5:  # minimum 0.5m between keyframes
                filtered_mask[idx] = True
                last_good = idx

    n_clean = filtered_mask.sum()
    print(f"  After cleaning: {n_clean} keyframes (removed {n - n_clean})")

    # Build clean FAISS index
    clean_embeddings = embeddings[filtered_mask]
    clean_poses = poses[filtered_mask]
    clean_ids = np.arange(n_clean)

    clean_index = faiss.IndexFlatIP(dim)
    clean_index.add(clean_embeddings)

    # Save cleaned map
    faiss.write_index(clean_index, os.path.join(CLEAN_DIR, "map_index.faiss"))
    np.save(os.path.join(CLEAN_DIR, "keyframe_poses.npy"), clean_poses)
    np.save(os.path.join(CLEAN_DIR, "keyframe_ids.npy"), clean_ids)
    print(f"  Saved cleaned map to {CLEAN_DIR}")

    return (clean_poses, clean_ids, clean_index, clean_embeddings,
            poses, ids, embeddings, filtered_mask)


def extract_bag_images(bag_path, topic='/camera/color/image_raw/compressed',
                       max_images=20, stride=None):
    """Extract sample compressed images from the bag."""
    conn = sqlite3.connect(bag_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return [], []
    topic_id = row[0]

    cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id = ?", (topic_id,))
    total = cursor.fetchone()[0]
    if stride is None:
        stride = max(1, total // max_images)

    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                   (topic_id,))

    images, timestamps = [], []
    for i, (ts, data) in enumerate(cursor):
        if i % stride != 0 or len(images) >= max_images:
            continue
        data_bytes = bytes(data)
        jpeg_start = data_bytes.find(b'\xff\xd8')
        if jpeg_start == -1:
            jpeg_start = data_bytes.find(b'\x89PNG')
        if jpeg_start == -1:
            continue
        np_arr = np.frombuffer(data_bytes[jpeg_start:], dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            timestamps.append(ts)

    conn.close()
    print(f"Extracted {len(images)} sample images from bag")
    return images, timestamps


def run_localization(images, index, poses, ids):
    """Test localization with SigLIP + FAISS."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP 2 (device={device})...")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).eval()

    results = []
    for i, img in enumerate(images):
        pil_img = PilImage.fromarray(img)
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.vision_model(**inputs).pooler_output
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        emb_np = emb.cpu().numpy().astype(np.float32)
        dists, idxs = index.search(emb_np, 5)

        best_idx = idxs[0][0]
        results.append({
            'query_idx': i,
            'best_kf_id': int(ids[best_idx]),
            'best_score': float(dists[0][0]),
            'best_pose': poses[best_idx],
            'top5_scores': dists[0].tolist(),
            'top5_indices': idxs[0].tolist(),
        })
        p = poses[best_idx]
        print(f"  Q{i:2d} -> KF {ids[best_idx]:3d} (score={dists[0][0]:.4f}) "
              f"at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    return results


def plot_all(clean_poses, clean_embeddings, clean_ids, images, results,
             orig_poses, orig_filtered_mask):
    """Generate all visualizations."""

    # ── Fig 1: Cleaned trajectory ──
    fig = plt.figure(figsize=(18, 7), facecolor='#1a1a2e')
    fig.suptitle("Cleaned Hybrid Map — Trajectory & Localization",
                 fontsize=15, fontweight='bold', color='white', y=0.98)

    x, y, z = clean_poses[:, 0], clean_poses[:, 1], clean_poses[:, 2]
    n_kf = len(clean_poses)
    c = np.linspace(0, 1, n_kf)

    # 3D
    ax1 = fig.add_subplot(131, projection='3d', facecolor='#1a1a2e')
    ax1.plot(x, y, z, color='#555577', lw=0.8, alpha=0.5)
    ax1.scatter(x, y, z, c=c, cmap='viridis', s=18, alpha=0.9)
    ax1.scatter(x[0], y[0], z[0], color='lime', s=100, marker='o', label='Start', edgecolors='w')
    ax1.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='X', label='End', edgecolors='w')
    ax1.set_title('3D Trajectory (cleaned)', color='white', fontsize=11)
    ax1.set_xlabel('X [m]', color='white', fontsize=8)
    ax1.set_ylabel('Y [m]', color='white', fontsize=8)
    ax1.set_zlabel('Z [m]', color='white', fontsize=8)
    ax1.legend(fontsize=7, facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    ax1.tick_params(colors='#aaa', labelsize=7)
    for p in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        p.fill = False

    # 2D bird's eye with localization results
    ax2 = fig.add_subplot(132, facecolor='#16213e')
    ax2.plot(x, y, color='#555577', lw=0.8, alpha=0.5)
    sc = ax2.scatter(x, y, c=np.arange(n_kf), cmap='viridis', s=15, alpha=0.8)
    ax2.scatter(x[0], y[0], color='lime', s=100, marker='o', edgecolors='w', zorder=5)
    ax2.scatter(x[-1], y[-1], color='red', s=100, marker='X', edgecolors='w', zorder=5)

    # Overlay localization results
    if results:
        mp = np.array([r['best_pose'] for r in results])
        ms = np.array([r['best_score'] for r in results])
        ax2.scatter(mp[:, 0], mp[:, 1], c=ms, cmap='RdYlGn', s=60, edgecolors='white',
                    lw=1.2, zorder=10, vmin=0.5, vmax=1.0, marker='D')
        for i, r in enumerate(results):
            ax2.annotate(f"Q{i}", (r['best_pose'][0], r['best_pose'][1]),
                         fontsize=5, color='white', ha='center', va='bottom',
                         xytext=(0, 4), textcoords='offset points')

    cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label('Keyframe Index', color='white', fontsize=8)
    cbar.ax.tick_params(colors='#aaa', labelsize=7)
    ax2.set_title("Bird's Eye + Localization Hits", color='white', fontsize=11)
    ax2.set_xlabel('X [m]', color='white', fontsize=9)
    ax2.set_ylabel('Y [m]', color='white', fontsize=9)
    ax2.set_aspect('equal')
    ax2.tick_params(colors='#aaa', labelsize=7)

    # Stats panel
    ax3 = fig.add_subplot(133, facecolor='#16213e')
    ax3.axis('off')
    diffs = np.linalg.norm(np.diff(clean_poses[:, :3], axis=0), axis=1)
    scores = [r['best_score'] for r in results] if results else [0]
    stats = (
        f"CLEANED MAP STATISTICS\n"
        f"{'='*30}\n"
        f"Keyframes:      {n_kf}\n"
        f"Embedding dim:  {clean_embeddings.shape[1]}\n"
        f"Path length:    {diffs.sum():.1f} m\n"
        f"X range:        [{x.min():.1f}, {x.max():.1f}]\n"
        f"Y range:        [{y.min():.1f}, {y.max():.1f}]\n"
        f"Z range:        [{z.min():.1f}, {z.max():.1f}]\n"
        f"\nORIGINAL MAP\n"
        f"{'='*30}\n"
        f"Total KFs:      {len(orig_poses)}\n"
        f"Removed:        {len(orig_poses) - n_kf}\n"
        f"  (diverged + duplicates)\n"
        f"\nLOCALIZATION TEST\n"
        f"{'='*30}\n"
        f"Queries:        {len(results)}\n"
        f"Mean score:     {np.mean(scores):.4f}\n"
        f"Min score:      {np.min(scores):.4f}\n"
        f"Max score:      {np.max(scores):.4f}\n"
        f"\nMAP COMPRESSION\n"
        f"{'='*30}\n"
        f"FAISS index:    {n_kf * clean_embeddings.shape[1] * 4 / 1024:.1f} KB\n"
        f"Poses:          {n_kf * 7 * 8 / 1024:.1f} KB\n"
        f"Total map:      {(n_kf * clean_embeddings.shape[1] * 4 + n_kf * 7 * 8) / 1024:.1f} KB"
    )
    ax3.text(0.05, 0.95, stats, transform=ax3.transAxes, fontsize=9,
             color='#ddd', family='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.9, edgecolor='#444'))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "05_cleaned_map_overview.png")
    plt.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # ── Fig 2: Embedding analysis (similarity + t-SNE) ──
    n = len(clean_embeddings)
    sim = clean_embeddings @ clean_embeddings.T

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a2e')
    fig.suptitle("Embedding Analysis (Cleaned Map)", fontsize=15,
                 fontweight='bold', color='white', y=0.98)

    im = axes[0].imshow(sim, cmap='inferno', vmin=0.5, vmax=1.0, aspect='auto')
    axes[0].set_title('Pairwise Cosine Similarity', color='white', fontsize=11)
    axes[0].set_xlabel('Keyframe Index', color='white', fontsize=9)
    axes[0].set_ylabel('Keyframe Index', color='white', fontsize=9)
    axes[0].tick_params(colors='#aaa', labelsize=7)
    cbar = plt.colorbar(im, ax=axes[0], pad=0.02)
    cbar.set_label('Cosine Similarity', color='white', fontsize=8)
    cbar.ax.tick_params(colors='#aaa', labelsize=7)

    # t-SNE
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    perp = min(30, n - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    emb_2d = tsne.fit_transform(clean_embeddings)
    sc = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(n), cmap='viridis',
                     s=20, alpha=0.9)
    ax2.scatter(emb_2d[0, 0], emb_2d[0, 1], color='lime', s=80, marker='o',
                edgecolors='w', zorder=5, label='Start')
    ax2.scatter(emb_2d[-1, 0], emb_2d[-1, 1], color='red', s=80, marker='X',
                edgecolors='w', zorder=5, label='End')
    cbar2 = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar2.set_label('Keyframe Index', color='white', fontsize=8)
    cbar2.ax.tick_params(colors='#aaa', labelsize=7)
    ax2.set_title('t-SNE of SigLIP Embeddings', color='white', fontsize=11)
    ax2.legend(fontsize=7, facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    ax2.tick_params(colors='#aaa', labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "06_embedding_analysis_clean.png")
    plt.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    # ── Fig 3: Query images with matches ──
    if images and results:
        n_show = min(len(images), 16)
        cols = 4
        rows = (n_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), facecolor='#1a1a2e')
        fig.suptitle("Localization: Query Images with Matched Keyframe Poses (Cleaned Map)",
                     fontsize=13, fontweight='bold', color='white', y=0.99)

        for i in range(rows * cols):
            ax = axes.flat[i]
            ax.set_facecolor('#16213e')
            if i >= n_show:
                ax.axis('off')
                continue
            ax.imshow(images[i])
            r = results[i]
            p = r['best_pose']
            ax.set_title(f"KF {r['best_kf_id']} | score={r['best_score']:.3f}\n"
                         f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})",
                         color='#00ff00', fontsize=8)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(OUT_DIR, "07_localization_clean.png")
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")

    # ── Fig 4: Score distribution ──
    if results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        scores = [r['best_score'] for r in results]
        ax.bar(range(len(scores)), scores, color='#4a90d9', edgecolor='#222', alpha=0.9)
        ax.axhline(np.mean(scores), color='lime', ls='--', lw=1.5,
                    label=f'Mean: {np.mean(scores):.4f}')
        ax.set_xlabel('Query Image Index', color='white', fontsize=10)
        ax.set_ylabel('Cosine Similarity Score', color='white', fontsize=10)
        ax.set_title('Localization Match Scores (Cleaned Map)',
                     color='white', fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9, facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
        ax.tick_params(colors='#aaa', labelsize=8)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, "08_score_distribution.png")
        plt.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")


def main():
    print("=" * 70)
    print("  Hybrid Localization — Clean Map & Re-Test")
    print("=" * 70)

    # Step 1: Clean map
    print("\n--- Step 1: Cleaning Map ---")
    (clean_poses, clean_ids, clean_index, clean_emb,
     orig_poses, orig_ids, orig_emb, filt_mask) = load_and_clean_map()

    # Step 2: Extract images
    print("\n--- Step 2: Extract Images ---")
    images, timestamps = extract_bag_images(BAG_DB, max_images=20)

    # Step 3: Run localization on cleaned map
    print("\n--- Step 3: Localization Test (Cleaned Map) ---")
    results = run_localization(images, clean_index, clean_poses, clean_ids)

    # Step 4: Visualize everything
    print("\n--- Step 4: Generating Visualizations ---")
    plot_all(clean_poses, clean_emb, clean_ids, images, results,
             orig_poses, filt_mask)

    # Final summary
    scores = [r['best_score'] for r in results]
    print("\n" + "=" * 70)
    print("  FINAL RESULTS (Cleaned Map)")
    print("=" * 70)
    print(f"  Keyframes:       {len(clean_poses)} (was {len(orig_poses)})")
    print(f"  Embedding dim:   {clean_emb.shape[1]}")
    diffs = np.linalg.norm(np.diff(clean_poses[:, :3], axis=0), axis=1)
    print(f"  Path length:     {diffs.sum():.1f} m")
    print(f"  Map size:        {(len(clean_poses) * clean_emb.shape[1] * 4 + len(clean_poses) * 7 * 8) / 1024:.1f} KB")
    print(f"  Queries:         {len(results)}")
    print(f"  Mean score:      {np.mean(scores):.4f}")
    print(f"  Min score:       {np.min(scores):.4f}")
    print(f"  Max score:       {np.max(scores):.4f}")
    print(f"\n  All visualizations in: {OUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
