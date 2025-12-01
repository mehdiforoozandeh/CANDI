#!/usr/bin/env python3
"""
EIC Benchmark Visualization Script

Compare CANDI imputation performance against EIC paper benchmark results.
Generates boxplots and parity plots for performance comparison.

Author: CANDI Team
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Mapping dictionaries for EIC paper codes
ASSAY_ID = {
    'M01': 'ATAC-seq',
    'M02': 'DNase-seq',
    'M03': 'H2AFZ',
    'M04': 'H2AK5ac',
    'M05': 'H2AK9ac',
    'M06': 'H2BK120ac',
    'M07': 'H2BK12ac',
    'M08': 'H2BK15ac',
    'M09': 'H2BK20ac',
    'M10': 'H2BK5ac',
    'M11': 'H3F3A',
    'M12': 'H3K14ac',
    'M13': 'H3K18ac',
    'M14': 'H3K23ac',
    'M15': 'H3K23me2',
    'M16': 'H3K27ac',
    'M17': 'H3K27me3',
    'M18': 'H3K36me3',
    'M19': 'H3K4ac',
    'M20': 'H3K4me1',
    'M21': 'H3K4me2',
    'M22': 'H3K4me3',
    'M23': 'H3K56ac',
    'M24': 'H3K79me1',
    'M25': 'H3K79me2',
    'M26': 'H3K9ac',
    'M27': 'H3K9me1',
    'M28': 'H3K9me2',
    'M29': 'H3K9me3',
    'M30': 'H3T11ph',
    'M31': 'H4K12ac',
    'M32': 'H4K20me1',
    'M33': 'H4K5ac',
    'M34': 'H4K8ac',
    'M35': 'H4K91ac'
}

CELL_TYPE_ID = {
    'C01': 'adipose_tissue',
    'C02': 'adrenal_gland',
    'C03': 'adrenalglandembryonic',
    'C04': 'amnion',
    'C05': 'BE2C',
    'C06': 'brainmicrovascularendothelial_cell',
    'C07': 'Caco-2',
    'C08': 'cardiac_fibroblast',
    'C09': 'CD4-positivealpha-betamemoryTcell',
    'C10': 'chorion',
    'C11': 'dermismicrovascularlymphaticvesselendothelial_cell',
    'C12': 'DND-41',
    'C13': 'endocrine_pancreas',
    'C14': 'ES-I3',
    'C15': 'G401',
    'C16': 'GM06990',
    'C17': 'H1',
    'C18': 'H9',
    'C19': 'HAP-1',
    'C20': 'heartleftventricle',
    'C21': 'hematopoieticmultipotentprogenitor_cell',
    'C22': 'HL-60',
    'C23': 'IMR-90',
    'C24': 'K562',
    'C25': 'KMS-11',
    'C26': 'lowerlegskin',
    'C27': 'mesenchymalstemcell',
    'C28': 'MG63',
    'C29': 'myoepithelialcellofmammarygland',
    'C30': 'NCI-H460',
    'C31': 'NCI-H929',
    'C32': 'neuralstemprogenitor_cell',
    'C33': 'occipital_lobe',
    'C34': 'OCI-LY7',
    'C35': 'omentalfatpad',
    'C36': 'peripheralbloodmononuclear_cell',
    'C37': 'prostate',
    'C38': 'RWPE2',
    'C39': 'SJCRH30',
    'C40': 'SJSA1',
    'C41': 'SK-MEL-5',
    'C42': 'skin_fibroblast',
    'C43': 'skinofbody',
    'C44': 'T47D',
    'C45': 'testis',
    'C46': 'trophoblast_cell',
    'C47': 'upperlobeofleftlung',
    'C48': 'urinary_bladder',
    'C49': 'uterus',
    'C50': 'vagina',
    'C51': 'WERI-Rb-1'
}

TEAM_ID = {
    0: 'Avocado',
    100: 'Average',
    3393417: 'Hongyang Li and Yuanfang Guan v1',
    3393574: 'Lavawizard',
    3393847: 'Guacamole',
    3393457: 'imp'
}

# Top performers for default filtering
TOP_PERFORMERS = [
    'Avocado',
    'Average',
    'Hongyang Li and Yuanfang Guan v1',
    'Lavawizard',
    'Guacamole',
    'imp'
]


def load_candi_metrics(metrics_path: str) -> pd.DataFrame:
    """
    Load CANDI metrics CSV and prepare for comparison.
    
    Args:
        metrics_path: Path to CANDI metrics CSV
        
    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(metrics_path)
    
    # Filter for imputed comparisons only
    df = df[df['comparison'] == 'imputed'].reset_index(drop=True)
    
    # Rename columns to match EIC format
    df = df.rename(columns={
        'bios': 'cell',
        'assay': 'assay',
        'P_MSE-GW': 'mse',
        'P_Pearson-GW': 'gwcorr',
        'P_Spearman-GW': 'gwspear'
    })
    
    # Standardize cell names (remove B_ prefix if present)
    df['cell'] = df['cell'].str.replace('B_', '', regex=False)
    
    # Add team identifier
    df['team'] = 'CANDI'
    
    return df[['team', 'cell', 'assay', 'mse', 'gwcorr', 'gwspear']]


def load_eic_benchmark(benchmark_path: str, use_team_id: bool = True) -> pd.DataFrame:
    """
    Load EIC benchmark CSV and prepare for comparison.
    
    Args:
        benchmark_path: Path to EIC benchmark CSV
        use_team_id: Whether to map team_id to team names
        
    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(benchmark_path)
    
    # Filter for bootstrap_id == 1
    df = df[df['bootstrap_id'] == 1].reset_index(drop=True)
    
    # Map codes to names if using team_id
    if use_team_id and 'team_id' in df.columns:
        df['team'] = df['team_id'].map(TEAM_ID)
        df = df.dropna(subset=['team'])  # Remove teams not in mapping
    
    # Map cell and assay codes to names
    df['cell'] = df['cell'].map(CELL_TYPE_ID)
    df['assay'] = df['assay'].map(ASSAY_ID)
    
    # Drop rows with unmapped values
    df = df.dropna(subset=['cell', 'assay'])
    
    return df[['team', 'cell', 'assay', 'mse', 'gwcorr', 'gwspear']]


def merge_datasets(candi_df: pd.DataFrame, eic_df: pd.DataFrame, 
                   all_teams: bool = False) -> pd.DataFrame:
    """
    Merge CANDI and EIC datasets.
    
    Args:
        candi_df: CANDI metrics DataFrame
        eic_df: EIC benchmark DataFrame
        all_teams: Include all teams or only top performers
        
    Returns:
        Merged DataFrame
    """
    # Filter EIC teams if needed
    if not all_teams:
        eic_df = eic_df[eic_df['team'].isin(TOP_PERFORMERS)].reset_index(drop=True)
    
    # Combine datasets
    merged_df = pd.concat([candi_df, eic_df], ignore_index=True)
    
    return merged_df


def comparison_boxplot(merged_df: pd.DataFrame, metric: str, 
                      output_path: Path, comparison_name: str):
    """
    Create boxplot with scatter overlay comparing teams across assays.
    
    Args:
        merged_df: Merged DataFrame with all teams
        metric: Metric to plot ('mse', 'gwcorr', 'gwspear')
        output_path: Path to save plot
        comparison_name: Name of comparison (e.g., "RAW Blind")
    """
    # Get unique teams (CANDI first, then others alphabetically)
    teams = ['CANDI'] + sorted([t for t in merged_df['team'].unique() if t != 'CANDI'])
    n_teams = len(teams)
    
    # Generate colormap
    if n_teams <= 8:
        color_palette = plt.get_cmap('Dark2')
        colors = [color_palette(i) for i in np.linspace(0, 1, 8)]
    else:
        color_palette = plt.get_cmap('tab20')
        colors = [color_palette(i) for i in np.linspace(0, 1, 20)]
    
    team_colors = {team: colors[i % len(colors)] for i, team in enumerate(teams)}
    
    # Get unique assays and sort by median performance
    assay_medians = merged_df.groupby('assay')[metric].median().sort_values(ascending=False)
    sorted_assays = assay_medians.index.tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, assay in enumerate(sorted_assays):
        assay_data = merged_df[merged_df['assay'] == assay]
        
        # Prepare boxplot data for each team
        boxplot_data = [
            assay_data[assay_data['team'] == team][metric].dropna()
            for team in teams
        ]
        
        # Calculate positions
        positions = [i * (n_teams + 1) + j for j in range(n_teams)]
        
        # Create boxplot
        bp = ax.boxplot(
            boxplot_data,
            positions=positions,
            patch_artist=True,
            widths=0.6,
            showcaps=False,
            showfliers=False,
            boxprops=dict(linewidth=1.2),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(linewidth=1.2)
        )
        
        # Color boxes and add scatter overlay
        for box_idx, (team, box) in enumerate(zip(teams, bp['boxes'])):
            color = team_colors[team]
            box.set_facecolor(color)
            box.set_alpha(0.6)
            
            # Add scatter overlay with jitter
            team_data = assay_data[assay_data['team'] == team][metric].dropna()
            if len(team_data) > 0:
                np.random.seed(hash(f"{metric}_{assay}_{team}") % (2**32))
                jitter = np.random.normal(0, 0.05, len(team_data))
                x_coords = positions[box_idx] + jitter
                ax.scatter(x_coords, team_data, color=color, alpha=0.6, s=20, zorder=10)
    
    # Formatting
    ax.set_xlabel('Assay', fontsize=14)
    ax.set_ylabel(metric.upper() if metric != 'mse' else 'MSE', fontsize=14)
    
    # Set x-ticks
    xticks_positions = [
        i * (n_teams + 1) + n_teams / 2 - 0.5
        for i in range(len(sorted_assays))
    ]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(sorted_assays, rotation=90, fontsize=12)
    
    # Add vertical separators
    for i in range(len(sorted_assays) - 1):
        ax.axvline(x=(i + 1) * (n_teams + 1) - 1, color='k', linestyle='--', linewidth=0.5)
    
    # Log scale for MSE
    if metric == 'mse':
        ax.set_yscale('log')
    
    # Legend
    handles = [
        plt.Line2D([0], [0], color=team_colors[team], lw=4, label=team)
        for team in teams
    ]
    ax.legend(
        handles=handles,
        loc='upper center',
        ncol=min(8, n_teams),
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        fontsize=12
    )
    
    # Add title with extra space
    ax.set_title(f'{comparison_name}: {metric.upper()}', fontsize=16, pad=40)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved boxplot: {output_path}")


def parity_multipanel(merged_df: pd.DataFrame, metric: str,
                     output_path: Path, comparison_name: str):
    """
    Create multi-panel parity plot with one subplot per assay.
    
    Args:
        merged_df: Merged DataFrame with all teams
        metric: Metric to plot ('mse', 'gwcorr', 'gwspear')
        output_path: Path to save plot
        comparison_name: Name of comparison (e.g., "RAW Blind")
    """
    # Get CANDI data
    candi_df = merged_df[merged_df['team'] == 'CANDI'].copy()
    candi_df = candi_df.set_index(['cell', 'assay'])
    
    # Get competitor teams
    competitor_teams = sorted([t for t in merged_df['team'].unique() if t != 'CANDI'])
    
    # Team styles: colors and markers
    team_styles = {}
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '8', '*', '+']
    colors_palette = plt.get_cmap('tab10')
    
    for i, team in enumerate(competitor_teams):
        team_styles[team] = {
            'marker': markers[i % len(markers)],
            'color': colors_palette(i % 10)
        }
    
    # Get assays with data
    assays_with_data = sorted(candi_df.index.get_level_values('assay').unique())
    n_assays = len(assays_with_data)
    
    if n_assays == 0:
        print(f"Warning: No assays with data for {metric}")
        return
    
    # Calculate grid dimensions
    n_cols = min(6, n_assays)
    n_rows = int(np.ceil(n_assays / n_cols))
    
    # Create figure
    subplot_size = 4
    fig_width = subplot_size * n_cols
    fig_height = subplot_size * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                            sharex=False, sharey=False, squeeze=False)
    axes = axes.flatten()
    
    # Track global min/max for consistent axes
    global_min, global_max = None, None
    
    # Plot each assay
    for idx, assay in enumerate(assays_with_data):
        ax = axes[idx]
        
        # Get CANDI data for this assay
        candi_assay = candi_df.xs(assay, level='assay')[metric]
        
        if candi_assay.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(assay, fontsize=10)
            continue
        
        # Track min/max
        if global_min is None:
            global_min = candi_assay.min()
            global_max = candi_assay.max()
        else:
            global_min = min(global_min, candi_assay.min())
            global_max = max(global_max, candi_assay.max())
        
        # Plot each competitor team
        for team in competitor_teams:
            team_data = merged_df[(merged_df['team'] == team) & 
                                 (merged_df['assay'] == assay)].copy()
            team_data = team_data.set_index('cell')[metric]
            
            # Find common cells
            common_cells = candi_assay.index.intersection(team_data.index)
            
            if len(common_cells) == 0:
                continue
            
            x_vals = team_data.loc[common_cells]
            y_vals = candi_assay.loc[common_cells]
            
            # Update global min/max
            global_min = min(global_min, x_vals.min())
            global_max = max(global_max, x_vals.max())
            
            style = team_styles[team]
            ax.scatter(
                x_vals, y_vals,
                label=team,
                marker=style['marker'],
                color=style['color'],
                alpha=0.7,
                s=50,
                edgecolor='none'
            )
        
        ax.set_title(assay, fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set consistent axes and add diagonal lines
    if global_min is not None:
        # Use log scale for MSE
        if metric == 'mse':
            # Set limits with some padding in log space
            log_min = max(0.001, global_min * 0.5)
            log_max = global_max * 2
            
            for idx in range(len(assays_with_data)):
                ax = axes[idx]
                ax.set_xscale('log')
                ax.set_yscale('log')
                # Draw diagonal line manually for log scale
                ax.plot([log_min, log_max], [log_min, log_max], 
                       color='grey', linestyle='--', alpha=0.7, linewidth=1.5, zorder=1)
                ax.set_xlim(log_min, log_max)
                ax.set_ylim(log_min, log_max)
                ax.set_aspect('equal', adjustable='box')
        else:
            pad = 0.05 * (global_max - global_min)
            lo, hi = global_min - pad, global_max + pad
            
            for idx in range(len(assays_with_data)):
                ax = axes[idx]
                ax.axline((0, 0), slope=1, color='grey', linestyle='--', alpha=0.7, linewidth=1.5)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect('equal', adjustable='box')
    
    # Remove empty subplots
    for idx in range(len(assays_with_data), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add shared labels
    fig.text(0.5, 0.02, f'EIC Team {metric.upper()}', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, f'CANDI {metric.upper()}', va='center', rotation='vertical',
            fontsize=14, fontweight='bold')
    
    # Create shared legend
    handles, labels = [], []
    for idx in range(min(len(assays_with_data), len(axes))):
        for h, l in zip(*axes[idx].get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    
    if handles:
        fig.legend(handles, labels, loc='upper center',
                  bbox_to_anchor=(0.5, 0.98),
                  ncol=min(6, len(competitor_teams)),
                  fontsize=11, frameon=True)
    
    plt.suptitle(f'{comparison_name}: CANDI vs Competitors ({metric.upper()})',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parity plot: {output_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="EIC Benchmark Visualization: Compare CANDI vs EIC paper results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with top performers only
  python eval_scripts/viz_eic_bench.py --candi-metrics models/my_model/preds/metrics.csv --output-dir results/eic_bench/

  # Compare with all teams
  python eval_scripts/viz_eic_bench.py --candi-metrics models/my_model/preds/metrics.csv --output-dir results/eic_bench/ --all-teams
        """
    )
    
    parser.add_argument('--candi-metrics', type=str, required=True,
                       help='Path to CANDI metrics CSV file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--all-teams', action='store_true',
                       help='Include all teams (default: top performers only)')
    parser.add_argument('--eic-paper-dir', type=str, default='eic_paper',
                       help='Directory containing EIC paper CSV files (default: eic_paper)')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eic_paper_dir = Path(args.eic_paper_dir)
    if not eic_paper_dir.exists():
        print(f"Error: EIC paper directory not found: {eic_paper_dir}")
        sys.exit(1)
    
    # Load CANDI metrics
    print("Loading CANDI metrics...")
    try:
        candi_df = load_candi_metrics(args.candi_metrics)
        print(f"Loaded {len(candi_df)} CANDI results")
    except Exception as e:
        print(f"Error loading CANDI metrics: {e}")
        sys.exit(1)
    
    # Process each benchmark comparison
    comparisons = [
        ('raw_blind', eic_paper_dir / '13059_2023_2915_MOESM2_ESM.csv', 'RAW Blind'),
        ('qnorm_blind', eic_paper_dir / '13059_2023_2915_MOESM3_ESM.csv', 'QNorm Blind'),
        ('qnorm_reprocessed_blind', eic_paper_dir / '13059_2023_2915_MOESM4_ESM.csv', 'QNorm Reprocessed Blind')
    ]
    
    for comp_id, benchmark_path, comp_name in comparisons:
        print(f"\n{'='*60}")
        print(f"Processing: {comp_name}")
        print(f"{'='*60}")
        
        if not benchmark_path.exists():
            print(f"Warning: Benchmark file not found: {benchmark_path}")
            continue
        
        # Load EIC benchmark
        print(f"Loading EIC benchmark from {benchmark_path.name}...")
        try:
            eic_df = load_eic_benchmark(str(benchmark_path))
            print(f"Loaded {len(eic_df)} EIC results")
        except Exception as e:
            print(f"Error loading EIC benchmark: {e}")
            continue
        
        # Merge datasets
        print("Merging datasets...")
        merged_df = merge_datasets(candi_df, eic_df, all_teams=args.all_teams)
        print(f"Merged dataset: {len(merged_df)} total results")
        print(f"Teams: {sorted(merged_df['team'].unique())}")
        
        # Generate visualizations for each metric
        metrics = ['mse', 'gwcorr', 'gwspear']
        
        for metric in metrics:
            print(f"\nGenerating visualizations for {metric}...")
            
            # Boxplot
            boxplot_path = output_dir / f"eic_bench_{comp_id}_{metric}.svg"
            try:
                comparison_boxplot(merged_df, metric, boxplot_path, comp_name)
            except Exception as e:
                print(f"Error generating boxplot for {metric}: {e}")
                import traceback
                traceback.print_exc()
            
            # Parity plot
            parity_path = output_dir / f"eic_parity_{comp_id}_{metric}.svg"
            try:
                parity_multipanel(merged_df, metric, parity_path, comp_name)
            except Exception as e:
                print(f"Error generating parity plot for {metric}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

