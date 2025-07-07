import argparse
import logging
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -----------------------------------------------------------------------------
# Set random seeds for reproducibility
# -----------------------------------------------------------------------------
np.random.seed(42)
sns.set_style('whitegrid')

# Set better visual styles for plots
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Custom color palette for better visualization
custom_palette = sns.color_palette("viridis", 2)


def load_data(h5_path: str, sample_size: int = 5000) -> pd.DataFrame:
    """Load scalar data from HDF5 file into a single DataFrame with sampling.

    Args:
        h5_path (str): Path to the HDF5 dataset file.
        sample_size (int): Approximate number of samples to load in total.

    Returns:
        pd.DataFrame: Combined DataFrame for train and test with limited samples.
    """
    logging.info(f"Loading data from {h5_path} with target sample size {sample_size}")
    
    with h5py.File(h5_path, 'r') as f:
        # Get total size of available data
        train_size = len(f['train']['L'])
        test_size = len(f['test']['L']) if 'test' in f else 0
        total_size = train_size + test_size
        
        # Calculate sampling rate to get approximately sample_size records
        sample_rate = min(1.0, sample_size / total_size)
        train_samples = int(train_size * sample_rate)
        test_samples = int(test_size * sample_rate) if test_size > 0 else 0
        
        logging.info(f"Sampling {train_samples} from train and {test_samples} from test")
        
        records = []
        for group_name, sample_count in [('train', train_samples), ('test', test_samples)]:
            if group_name not in f:
                continue
                
            grp = f[group_name]
            # Get random indices for sampling
            indices = np.random.choice(len(grp['L']), size=sample_count, replace=False)
            
            for i in indices:
                # basic attributes
                rec = {key: grp[key][i] for key in (
                    'L', 'mu', 't', 'delta', 'disorder_strength',
                    'has_mzm', 'min_energy', 'edge_localization'
                )}
                # store index and group
                rec.update(group=group_name, index=i)
                records.append(rec)
                
    df = pd.DataFrame(records)
    logging.info(f"Loaded {len(df)} samples in total")
    return df


def plot_energy_spectrum(df: pd.DataFrame, f: h5py.File, num_samples: int, save_dir: str = None) -> None:
    """Plot sorted eigenvalues vs. index for samples with and without MZM."""
    logging.info("Plotting energy spectra")
    
    for cls, label in ((1, 'With MZM'), (0, 'Without MZM')):
        sub = df[df['has_mzm'] == cls]
        picked = sub.sample(min(num_samples, len(sub)), random_state=42)
        
        for _, row in picked.iterrows():
            spectra = f[row['group']]['spectra/eigenvalues'][row['index']]
            
            plt.figure(figsize=(10, 7))
            plt.plot(np.sort(spectra), 'o-', color=custom_palette[cls], 
                     linewidth=2, markersize=6, alpha=0.8)
            
            plt.title(f'Energy Spectrum ({label})', fontweight='bold')
            plt.suptitle(f'Min Energy: {row["min_energy"]:.4f}', fontsize=12)
            plt.xlabel('Eigenvalue Index')
            plt.ylabel('Eigenvalue')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add parameter details in a text box
            param_text = (f"L={row['L']}, μ={row['mu']:.2f}, t={row['t']:.2f}\n"
                          f"Δ={row['delta']:.2f}, disorder={row['disorder_strength']:.2f}")
            plt.figtext(0.02, 0.02, param_text, fontsize=10)
            
            plt.tight_layout()
            
            if save_dir:
                fname = os.path.join(save_dir, f"spectrum_{label}_{row['group']}_{row['index']}.png")
                plt.savefig(fname, bbox_inches='tight', dpi=300)
            plt.show()


def plot_min_energy_histogram(df: pd.DataFrame, save_dir: str = None) -> None:
    """Overlayed histograms of min_energy for has_mzm=0 and has_mzm=1."""
    logging.info("Plotting minimum energy histograms")
    
    plt.figure(figsize=(10, 7))
    
    # Create custom labels with counts
    mzm_count = len(df[df['has_mzm'] == 1])
    no_mzm_count = len(df[df['has_mzm'] == 0])
    labels = [f'No MZM (n={no_mzm_count})', f'With MZM (n={mzm_count})']
    
    # Better visualization with enhanced histplot
    ax = sns.histplot(
        data=df, x='min_energy', hue='has_mzm', 
        kde=True, palette=custom_palette,
        element="step", alpha=0.6, bins=30,
        hue_order=[0, 1]
    )
    
    # Customize legend
    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title='Classification', loc='upper right')
    
    plt.title('Distribution of Minimum Energy by MZM Presence', fontweight='bold')
    plt.xlabel('Minimum Energy')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical line at a threshold value (e.g., 0.01) if meaningful
    plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.7, 
                label='Example Threshold')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "min_energy_hist.png"), 
                   bbox_inches='tight', dpi=300)
    plt.show()


def plot_edge_localization_vs_min_energy(df: pd.DataFrame, n_samples: int, save_dir: str = None) -> None:
    """Scatter plot of edge_localization vs. min_energy, colored by has_mzm."""
    logging.info("Plotting edge localization vs minimum energy")
    
    n0 = min(n_samples, len(df[df['has_mzm'] == 0]))
    n1 = min(n_samples, len(df[df['has_mzm'] == 1]))
    
    subset = pd.concat([
        df[df['has_mzm'] == 0].sample(n0, random_state=42),
        df[df['has_mzm'] == 1].sample(n1, random_state=42),
    ])
    
    plt.figure(figsize=(10, 7))
    
    # Enhanced scatter plot
    scatter = sns.scatterplot(
        data=subset, 
        x='min_energy', 
        y='edge_localization', 
        hue='has_mzm', 
        palette=custom_palette,
        s=80,  # Larger point size
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Customize legend
    scatter.legend(title='MZM Present', labels=['No', 'Yes'])
    
    plt.title('Edge Localization vs. Minimum Energy', fontweight='bold')
    plt.xlabel('Minimum Energy')
    plt.ylabel('Edge Localization')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add log scale for better visualization if data spans multiple orders of magnitude
    # Safely check ratio to avoid division by zero
    min_val = subset['min_energy'].min()
    max_val = subset['min_energy'].max()
    if min_val > 0 and max_val / min_val > 100:
        plt.xscale('log')
    elif min_val <= 0:
        # If we have zero or negative values, don't use log scale
        logging.info("Min energy contains zero or negative values, skipping log scale")
    
    # Add sample count as a subtitle
    plt.suptitle(f'Total samples: {len(subset)} ({n0} without MZM, {n1} with MZM)', 
                fontsize=12, y=0.95)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "edge_loc_vs_min_energy.png"), 
                   bbox_inches='tight', dpi=300)
    plt.show()


def plot_phase_diagram(df: pd.DataFrame, save_dir: str = None) -> None:
    """Hexbin phase diagram of mu vs. delta, colored by MZM frequency."""
    logging.info("Plotting phase diagram: μ vs. Δ")
    
    plt.figure(figsize=(11, 8))
    
    # Higher resolution hexbin with better colormap
    hb = plt.hexbin(
        df['mu'], df['delta'], C=df['has_mzm'],
        reduce_C_function=np.mean, gridsize=40, 
        cmap='viridis', alpha=0.95,
        linewidths=0.2
    )
    
    # Enhanced colorbar
    cbar = plt.colorbar(hb, label='MZM Frequency')
    cbar.ax.tick_params(size=0)  # Hide tick marks on colorbar
    
    # Theoretical phase boundary (if applicable - customize based on your model)
    # Example: |μ| < 2t is the topological phase for the Kitaev chain
    t_value = df['t'].mean()  # Using mean t value from data
    x = np.linspace(-4*t_value, 4*t_value, 100)
    plt.plot(x, np.zeros_like(x) + 0.5, 'r--', alpha=0.7, linewidth=2, 
             label='Example Phase Boundary')
    
    plt.title('Phase Diagram: Chemical Potential (μ) vs. Pairing Potential (Δ)', fontweight='bold')
    plt.xlabel('μ (Chemical Potential)')
    plt.ylabel('Δ (Pairing Potential)')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Add statistics as a subtitle
    mzm_percent = 100 * df['has_mzm'].mean()
    plt.suptitle(f'Dataset statistics: {len(df)} samples, {mzm_percent:.1f}% contain MZM', 
                fontsize=12, y=0.95)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "phase_diagram_mu_delta.png"), 
                   bbox_inches='tight', dpi=300)
    plt.show()


def plot_edge_localization_histogram(df: pd.DataFrame, save_dir: str = None) -> None:
    """Histogram of edge_localization with log-scale y-axis if skewed."""
    logging.info("Plotting edge localization histogram")
    
    plt.figure(figsize=(10, 7))
    
    # Enhanced histogram with KDE
    sns.histplot(
        data=df, 
        x='edge_localization', 
        bins=50, 
        kde=True,
        color=sns.color_palette("viridis")[5],
        line_kws={'linewidth': 2},
        alpha=0.7
    )
    
    plt.yscale('log')
    plt.title('Distribution of Edge Localization', fontweight='bold')
    plt.xlabel('Edge Localization')
    plt.ylabel('Count (log scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics
    mean_val = df['edge_localization'].mean()
    median_val = df['edge_localization'].median()
    
    stats_text = f"Mean: {mean_val:.4f}\nMedian: {median_val:.4f}"
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "edge_loc_hist.png"), 
                   bbox_inches='tight', dpi=300)
    plt.show()

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -----------------------------------------------------------------------------
# Set random seeds for reproducibility
# -----------------------------------------------------------------------------
np.random.seed(42)
sns.set_style('whitegrid')

# Set better visual styles for plots
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Custom color palette for better visualization
custom_palette = sns.color_palette("viridis", 2)

# Ensure Line2D is available for custom legend handles
from matplotlib.lines import Line2D

def get_mzm_handles():
    """Return legend handles for MZM absence (0) and presence (1)."""
    return [
        Line2D([0], [0], marker='o', color=custom_palette[0], linestyle='', markersize=8),
        Line2D([0], [0], marker='o', color=custom_palette[1], linestyle='', markersize=8)
    ]

# Approach 2: Feature extraction approach - extract meaningful features from each spectrum
# =============================================================================
def plot_pca_features(df: pd.DataFrame, f: h5py.File, num_samples: int, save_dir: str = None) -> None:
    """PCA using extracted features from spectra that don't depend on spectrum length."""
    logging.info("Performing PCA on extracted spectral features")
    
    n = num_samples // 2
    n0 = min(n, len(df[df['has_mzm'] == 0]))
    n1 = min(n, len(df[df['has_mzm'] == 1]))
    
    subset = pd.concat([
        df[df['has_mzm'] == 0].sample(n0, random_state=42),
        df[df['has_mzm'] == 1].sample(n1, random_state=42),
    ])
    
    # Extract features
    features = []
    for _, row in subset.iterrows():
        spectra = f[row['group']]['spectra/eigenvalues'][row['index']]
        sorted_s = np.sort(np.abs(spectra))
        features.append({
            'min_energy': row['min_energy'],
            'edge_localization': row['edge_localization'],
            'mean_energy': np.mean(spectra),
            'std_energy': np.std(spectra),
            'median_energy': np.median(spectra),
            'energy_gap': sorted_s[1] - sorted_s[0] if len(sorted_s) > 1 else 0,
            'kurtosis': np.mean((spectra - np.mean(spectra))**4) / (np.std(spectra)**4) if np.std(spectra) > 0 else 0,
            'skewness': np.mean((spectra - np.mean(spectra))**3) / (np.std(spectra)**3) if np.std(spectra) > 0 else 0,
            'lowest_5_mean': np.mean(sorted_s[:5]),
            'highest_5_mean': np.mean(sorted_s[-5:])
        })
    
    features_df = pd.DataFrame(features)
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(features_df)
    pca = PCA(n_components=2)
    result = pca.fit_transform(scaled)
    exp_var = pca.explained_variance_ratio_ * 100

    # Heatmap of feature contributions
    plt.figure(figsize=(12, 5))
    contrib = pd.DataFrame(
        pca.components_.T,
        index=features_df.columns,
        columns=[f'PC{i+1}' for i in range(2)]
    )
    sns.heatmap(contrib, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Feature Contributions to Principal Components', fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_feature_contributions.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # PCA scatter plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=result[:, 0], y=result[:, 1],
        hue=subset['has_mzm'], palette=custom_palette,
        s=100, alpha=0.7, edgecolor='w', linewidth=0.5
    )
    plt.legend(handles=get_mzm_handles(), labels=['No', 'Yes'], title='MZM Present')
    plt.title('PCA of Extracted Spectral Features', fontweight='bold')
    plt.xlabel(f'PC1 ({exp_var[0]:.1f}% var)')
    plt.ylabel(f'PC2 ({exp_var[1]:.1f}% var)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.suptitle(f'Total samples: {len(subset)} ({n0} no, {n1} yes)', fontsize=12, y=0.95)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_spectral_features.png"), dpi=300, bbox_inches='tight')
    plt.show()

# Approach 3: Bin the spectra into histograms with fixed bins
# =============================================================================
def plot_pca_binned(df: pd.DataFrame, f: h5py.File, num_samples: int, save_dir: str = None, num_bins: int = 50) -> None:
    """PCA using binned histograms of the eigenvalue spectra."""
    logging.info(f"Performing PCA on binned spectra with {num_bins} bins")
    n = num_samples // 2
    n0 = min(n, len(df[df['has_mzm'] == 0]))
    n1 = min(n, len(df[df['has_mzm'] == 1]))
    subset = pd.concat([
        df[df['has_mzm'] == 0].sample(n0, random_state=42),
        df[df['has_mzm'] == 1].sample(n1, random_state=42),
    ])
    
    # Determine global range for binning
    gmin, gmax = np.inf, -np.inf
    for _, row in subset.iterrows():
        vals = f[row['group']]['spectra/eigenvalues'][row['index']]
        gmin, gmax = min(gmin, np.min(vals)), max(gmax, np.max(vals))
    padding = 0.05 * (gmax - gmin)
    bins = np.linspace(gmin - padding, gmax + padding, num_bins + 1)
    
    hist_data = []
    for _, row in subset.iterrows():
        vals = f[row['group']]['spectra/eigenvalues'][row['index']]
        hist, _ = np.histogram(vals, bins=bins, density=True)
        hist_data.append(hist)
    arr = np.vstack(hist_data)

    pca = PCA(n_components=2)
    res = pca.fit_transform(arr)
    exp = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=res[:, 0], y=res[:, 1], hue=subset['has_mzm'], palette=custom_palette,
        s=100, alpha=0.7, edgecolor='w', linewidth=0.5
    )
    plt.legend(handles=get_mzm_handles(), labels=['No', 'Yes'], title='MZM Present')
    plt.title(f'PCA of Binned Spectra ({num_bins} bins)', fontweight='bold')
    plt.xlabel(f'PC1 ({exp[0]:.1f}% var)')
    plt.ylabel(f'PC2 ({exp[1]:.1f}% var)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.suptitle(f'Total samples: {len(subset)} ({n0} no, {n1} yes)', fontsize=12, y=0.95)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_binned_spectra.png"), dpi=300, bbox_inches='tight')
    plt.show()



def main() -> None:
    """Parse arguments and generate all requested plots."""
    parser = argparse.ArgumentParser(description='Generate presentation plots.')
    parser.add_argument('h5_path', type=str, help='HDF5 dataset path')
    parser.add_argument('--n_spectrum', type=int, help='Samples per class for spectra', default=3)
    parser.add_argument('--n_scatter', type=int, help='Samples per class for scatter', default=1000)
    parser.add_argument('--n_pca', type=int, help='Total samples for PCA', default=1000)
    parser.add_argument('--sample_size', type=int, help='Total samples to load from dataset', default=5000)
    parser.add_argument('--save_dir', type=str, help='Directory to save figures', default=None)
    parser.add_argument('--num_bins', type=int, help='Number of bins for binned PCA', default=50)
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        logging.info(f"Saving plots to {args.save_dir}")

    # Load limited number of samples
    df = load_data(args.h5_path, sample_size=args.sample_size)
    
    logging.info(f"Loaded {len(df)} samples: {len(df[df['has_mzm'] == 1])} with MZM, {len(df[df['has_mzm'] == 0])} without MZM")
    
    with h5py.File(args.h5_path, 'r') as f:
        plot_energy_spectrum(df, f, args.n_spectrum, args.save_dir)
        plot_min_energy_histogram(df, args.save_dir)
        plot_edge_localization_vs_min_energy(df, args.n_scatter, args.save_dir)
        plot_phase_diagram(df, args.save_dir)
        plot_edge_localization_histogram(df, args.save_dir)
        plot_pca_binned(df, f, args.n_pca, args.save_dir, args.num_bins)
        plot_pca_features(df, f, args.n_pca, args.save_dir)

if __name__ == '__main__':
    main()