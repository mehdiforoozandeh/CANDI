#!/usr/bin/env python3
"""
Visualize Assay Co-availability and Imputation Hop Distance.

This script computes the Conditional Availability Probability P(Target|Input)
and uses it to build a directed graph of assay connectivity. It then computes
the "Imputation Hop Distance" (shortest path length) between all pairs of assays.

Usage:
    python eval_scripts/viz_coavailability.py <metadata_csv> [--outdir <dir>]
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Try importing networkx, but fallback if missing
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

INCLUDED_ASSAYS = [
    'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac', 
    'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3', 
    'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac', 
    'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac'
]

def compute_coavailability(df):
    """
    Compute Conditional Availability Probability Matrix P(B|A).
    Returns:
        prob_matrix (pd.DataFrame): P(col|row)
        counts (pd.Series): Total counts per assay
        co_counts (pd.DataFrame): Raw co-occurrence counts
    """
    # Filter for included assays
    df = df[df['assay_name'].isin(INCLUDED_ASSAYS)].copy()
    
    # Group assays by biosample
    biosample_assays = df.groupby('biosample_name')['assay_name'].apply(set)
    
    assays = sorted(list(set(INCLUDED_ASSAYS))) # Ensure consistent order and specific set
    n = len(assays)
    assay_to_idx = {a: i for i, a in enumerate(assays)}
    
    # Initialize matrices
    co_mat = np.zeros((n, n), dtype=int)
    counts = np.zeros(n, dtype=int)
    
    # Count occurrences
    for assays_in_bio in biosample_assays:
        # Convert to indices
        indices = [assay_to_idx[a] for a in assays_in_bio if a in assay_to_idx]
        
        # Increment total counts
        for i in indices:
            counts[i] += 1
            
        # Increment co-occurrence counts (including self)
        for i in indices:
            for j in indices:
                co_mat[i, j] += 1
                
    # Compute probabilities P(j|i) = N(i,j) / N(i)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_mat = co_mat / counts[:, None]
        prob_mat = np.nan_to_num(prob_mat)
        
    prob_df = pd.DataFrame(prob_mat, index=assays, columns=assays)
    co_df = pd.DataFrame(co_mat, index=assays, columns=assays)
    counts_ser = pd.Series(counts, index=assays)
    
    return prob_df, counts_ser, co_df

def compute_hop_distance_numpy(prob_df, threshold=0.1):
    """
    Compute All-Pairs Shortest Path using Floyd-Warshall (numpy implementation).
    """
    assays = prob_df.index
    n = len(assays)
    adj = prob_df.values > threshold
    
    # Initialize distance matrix
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    
    # Direct edges have distance 1
    dist[adj] = 1
    np.fill_diagonal(dist, 0) # reset diagonal
    
    # Floyd-Warshall
    for k in range(n):
        dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
        
    dist_df = pd.DataFrame(dist, index=assays, columns=assays)
    return dist_df

def plot_heatmap(df, title, outpath, vmin=0, vmax=1, cmap="viridis", annot=False):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap=cmap, vmin=vmin, vmax=vmax, square=True, 
                xticklabels=True, yticklabels=True, annot=annot, fmt=".1f" if annot else "")
    plt.title(title)
    plt.xlabel("Target Assay (B)")
    plt.ylabel("Input Assay (A)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_network_simple(prob_df, counts_ser, threshold, title, outpath):
    """Manual network plot implementation without networkx."""
    plt.figure(figsize=(14, 14))
    ax = plt.gca()
    
    assays = prob_df.index
    n = len(assays)
    
    # Sort assays by count (availability) in descending order
    # so largest node is at 12 o'clock (top)
    counts = counts_ser.values
    sorted_indices = np.argsort(-counts)  # Descending order
    assays = assays[sorted_indices]
    counts = counts[sorted_indices]
    
    # Reorder prob_df rows and columns to match sorted assays
    prob_df = prob_df.loc[assays, assays]
    
    # Layout: Circular, starting at 12 o'clock (pi/2) going clockwise
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles = np.pi/2 - angles # Start at top (pi/2) and go clockwise
    
    radius_layout = 0.85
    x = radius_layout * np.cos(angles)
    y = radius_layout * np.sin(angles)
    
    # Normalize counts for node size (Area proportional to count -> Radius proportional to sqrt(count))
    max_count = counts.max()
    if max_count > 0:
        node_radii = 0.01 + 0.06 * np.sqrt(counts / max_count)
    else:
        node_radii = np.full(n, 0.02)
    
    # Draw edges first (so nodes are on top)
    adj = prob_df.values
    
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm
    
    # Get magma colormap
    cmap = cm.get_cmap('magma')
    
    # Pre-calculate node positions and radii for quick lookup
    node_params = list(zip(x, y, node_radii))
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            weight = adj[i, j]
            
            if weight > threshold:
                # Calculate start and end points at the edge of the circles
                xi, yi, ri = node_params[i]
                xj, yj, rj = node_params[j]
                
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > (ri + rj):
                    # Start at boundary of source
                    start_x = xi + (dx / dist) * ri
                    start_y = yi + (dy / dist) * ri
                    
                    # End at boundary of target (minus a small gap for arrow tip)
                    end_x = xj - (dx / dist) * rj
                    end_y = yj - (dy / dist) * rj
                    
                    # Line width proportional to probability
                    lw = 0.5 + 4.0 * weight
                    # Alpha also proportional, but kept high for visibility
                    alpha = 0.6 + 0.3 * weight
                    
                    # Create curved path with multiple segments for gradient
                    # Curvature to separate A->B from B->A
                    rad = 0.15 # radius of curvature
                    
                    # Calculate control point for quadratic bezier curve
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    
                    # Perpendicular offset for curvature
                    perp_x = -(end_y - start_y)
                    perp_y = (end_x - start_x)
                    perp_length = np.sqrt(perp_x**2 + perp_y**2)
                    if perp_length > 0:
                        perp_x /= perp_length
                        perp_y /= perp_length
                    
                    # Control point offset by rad
                    ctrl_x = mid_x + perp_x * rad
                    ctrl_y = mid_y + perp_y * rad
                    
                    # Sample points along curved path (quadratic bezier)
                    n_segments = 20
                    t = np.linspace(0, 1, n_segments + 1)
                    # Quadratic Bezier: B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
                    curve_x = (1-t)**2 * start_x + 2*(1-t)*t * ctrl_x + t**2 * end_x
                    curve_y = (1-t)**2 * start_y + 2*(1-t)*t * ctrl_y + t**2 * end_y
                    
                    # Create line segments with gradient colors
                    points = np.column_stack([curve_x, curve_y])
                    segments = np.stack([points[:-1], points[1:]], axis=1)
                    
                    # Color gradient from magma colormap
                    # Start with darker/lower value, end with brighter/higher value
                    colors = [cmap(weight * (0.3 + 0.7 * ti)) for ti in np.linspace(0, 1, n_segments)]
                    
                    lc = LineCollection(segments, colors=colors, linewidths=lw, alpha=alpha)
                    ax.add_collection(lc)
                    
                    # Add arrowhead at the end
                    # Calculate direction at the last segment
                    arrow_start_x = curve_x[-2]
                    arrow_start_y = curve_y[-2]
                    arrow_end_x = curve_x[-1]
                    arrow_end_y = curve_y[-1]
                    
                    arrow = FancyArrowPatch(
                        posA=(arrow_start_x, arrow_start_y), 
                        posB=(arrow_end_x, arrow_end_y),
                        arrowstyle='-|>',
                        mutation_scale=12,
                        linewidth=0,  # No line, just arrow head
                        color=cmap(weight),
                        alpha=alpha,
                        shrinkA=0, shrinkB=0
                    )
                    ax.add_patch(arrow)

    # Draw nodes
    for i in range(n):
        xi, yi, ri = node_params[i]
        
        # Grey nodes to emphasize colored edges
        circle = plt.Circle((xi, yi), ri, color='lightgrey', alpha=0.9, ec='darkgrey', linewidth=1.5)
        ax.add_patch(circle)
        
        # Label alignment - push label outwards
        label_dist = 1.05 * radius_layout + ri
        label_x = label_dist * np.cos(angles[i])
        label_y = label_dist * np.sin(angles[i])
        
        # Adjust alignment based on quadrant
        angle_deg = np.degrees(angles[i]) % 360
        if 90 < angle_deg < 270:
            ha = 'right'
            rotation = angle_deg + 180
        else:
            ha = 'left'
            rotation = angle_deg
            
        # Simplified horizontal alignment is often more readable
        if xi > 0.1: ha='left'
        elif xi < -0.1: ha='right'
        else: ha='center'
        
        va = 'center'
        if yi > 0.1: va='bottom'
        elif yi < -0.1: va='top'
        
        # Offset label slightly from node
        offset_scale = 1.1 + ri
        lx = xi * (1.0 + ri*2) 
        ly = yi * (1.0 + ri*2)
        
        # Use simpler radial placement
        dist_from_center = np.sqrt(xi**2 + yi**2)
        if dist_from_center > 0:
            dir_x = xi / dist_from_center
            dir_y = yi / dist_from_center
            text_x = xi + dir_x * (ri + 0.05)
            text_y = yi + dir_y * (ri + 0.05)
        else:
            text_x, text_y = xi, yi

        plt.text(text_x, text_y, assays[i], 
                 ha=ha, va=va, fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    plt.title(title, fontsize=16, y=1.02)
    
    # Add legends
    # 1. Size Legend
    sizes_legend_vals = np.linspace(counts.min(), counts.max(), 3)
    if len(sizes_legend_vals) > 0:
        legend_elements = []
        for val in sizes_legend_vals:
            if max_count > 0:
                r = 0.01 + 0.06 * np.sqrt(val / max_count)
            else:
                r = 0.02
            # Use scatter proxy
            legend_elements.append(plt.scatter([], [], s=(r*1000)**2, c='skyblue', edgecolors='steelblue', label=f"{int(val)}"))
        
        # Note: scatter size in points^2 vs circle radius in data units is tricky to match exactly without transforms
        # Simpler approach: Text description
        plt.text(-1.3, -1.3, f"Node Area $\propto$ Availability\n(Max: {int(counts.max())})", fontsize=10)

    # 2. Edge Legend
    plt.text(1.3, -1.3, "Edge Color & Thickness $\propto$ P(Target|Input)\n(magma colormap)", ha='right', fontsize=10)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze Assay Co-availability")
    parser.add_argument("csv", type=str, help="Metadata CSV path")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.csv))[0]
    
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    print("Computing co-availability matrix...")
    prob_df, counts, co_df = compute_coavailability(df)
    
    # Save raw matrices
    prob_df.to_csv(os.path.join(args.outdir, f"coavailability_prob_{base_name}.csv"))
    co_df.to_csv(os.path.join(args.outdir, f"coavailability_counts_{base_name}.csv"))
    
    # Sort assays by availability (descending order) for heatmaps
    sorted_assays = counts.sort_values(ascending=False).index
    prob_df_sorted = prob_df.loc[sorted_assays, sorted_assays]
    
    # Plot Probability Heatmap (sorted by availability)
    plot_heatmap(prob_df_sorted, 
                 f"Conditional Availability P(Target|Input) - {base_name}", 
                 os.path.join(args.outdir, f"heatmap_prob_{base_name}.png"),
                 vmin=0, vmax=1, cmap="Blues")
    
    # Compute and Plot Hop Distance for different thresholds
    thresholds = [0.1, 0.25, 0.5]
    
    for tau in thresholds:
        print(f"Computing Hop Distances (tau={tau})...")
        dist_df = compute_hop_distance_numpy(prob_df_sorted, threshold=tau)
        
        # Save distance matrix
        dist_df.to_csv(os.path.join(args.outdir, f"hop_distance_tau{tau}_{base_name}.csv"))
        
        # Plot Distance Heatmap (already sorted)
        dist_plot = dist_df.copy()
        max_dist = 4
        dist_plot[dist_plot > max_dist] = max_dist + 1 
        
        plt.figure(figsize=(12, 10))
        # Custom color map
        # 0 (self) -> White
        # 1 -> Green (Direct)
        # 2 -> Yellow
        # 3 -> Orange
        # 4 -> Red
        # 5 (Inf) -> Black/Gray
        cmap = matplotlib.colors.ListedColormap(['white', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#34495e'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        sns.heatmap(dist_plot, cmap=cmap, norm=norm, square=True, 
                    xticklabels=True, yticklabels=True, 
                    cbar_kws={"ticks": [0, 1, 2, 3, 4, 5], "label": "Hop Distance (Edges)"})
        
        plt.title(f"Imputation Hop Distance (Threshold > {tau}) - {base_name}")
        plt.xlabel("Target Assay (C)")
        plt.ylabel("Input Assay (A)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"heatmap_hops_tau{tau}_{base_name}.png"), dpi=150)
        plt.close()
        
        # Plot Network
        if HAS_NX:
            # Reconstruct graph for NX plotting
            G = nx.DiGraph()
            assays = prob_df_sorted.index
            G.add_nodes_from(assays)
            rows, cols = np.where(prob_df_sorted.values > tau)
            for r, c in zip(rows, cols):
                if r != c: G.add_edge(assays[r], assays[c])
            
            plt.figure(figsize=(12, 12))
            pos = nx.circular_layout(G)
            node_sizes = [np.log2(counts[a] + 1) * 100 for a in assays]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=True, arrowsize=10)
            nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
            plt.title(f"Co-availability Network (tau={tau}) - {base_name}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"network_tau{tau}_{base_name}.png"), dpi=150)
            plt.close()
        else:
            plot_network_simple(prob_df_sorted, counts, tau, 
                                f"Co-availability Network (tau={tau}) - {base_name}",
                                os.path.join(args.outdir, f"network_tau{tau}_{base_name}.png"))

    print("Done.")

if __name__ == "__main__":
    main()
