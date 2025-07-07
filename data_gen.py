#!/usr/bin/env python3
"""
Kitaev Chain Dataset Generator for ML-based MZM Detection (Using Kwant & CuPy)

This script generates a dataset of 1D Kitaev chain configurations using Kwant for quantum transport 
calculations and CuPy for GPU acceleration.

The dataset contains:
- 200000 training samples
- 30000 test samples
- Random non-overlapping parameter configurations between train and test sets
- ~40-60% of configurations exhibiting Majorana Zero Modes (MZMs)

Physical Background:
------------------
The 1D Kitaev chain is a model for a p-wave superconductor that can host Majorana
Zero Modes (MZMs) at its ends under certain parameter regimes. The model is 
described by a tight-binding Bogoliubov-de Gennes (BdG) Hamiltonian with particle-hole symmetry.

References:
- Kitaev, A. Y. (2001). "Unpaired Majorana fermions in quantum wires"
- Alicea, J. (2012). "New directions in the pursuit of Majorana fermions in solid state systems"
"""

import cupy as cp
import numpy as np
import kwant
import h5py
import time
import os
from datetime import datetime
from scipy import sparse

# Set random seed for reproducibility
RANDOM_SEED = 42
cp.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Dataset size configuration
N_TRAIN = 200000
N_TEST = 30000
N_TOTAL = N_TRAIN + N_TEST

# Physical parameters
class KitaevChainParameters:
    """Physical parameters for the Kitaev chain model with appropriate units."""
    def __init__(self):
        # System size (number of sites)
        self.L_min = 30
        self.L_max = 100
        
        # Chemical potential range [eV]
        self.mu_min = -4.0
        self.mu_max = 4.0
        
        # Hopping amplitude range [eV]
        self.t_min = 0.5
        self.t_max = 2.0
        
        # Superconducting pairing amplitude range [eV]
        self.delta_min = 0.2
        self.delta_max = 1.5
        
        # Disorder strength range [eV]
        self.disorder_min = 0.0
        self.disorder_max = 0.5
        
        # Energy threshold for identifying MZMs [eV]
        self.mzm_energy_threshold = 1e-6

def make_kitaev_kwant_system(L, t=1.0, mu=0.0, delta=1.0, disorder=None):
    """
    Build a 1D Kitaev chain using Kwant.
    
    Parameters:
    -----------
    L : int
        System size (number of sites)
    t : float
        Hopping amplitude [eV]
    mu : float
        Chemical potential [eV]
    delta : float
        Superconducting pairing amplitude [eV]
    disorder : np.ndarray or None
        On-site disorder potentials [eV]
        
    Returns:
    --------
    syst : kwant.Builder
        Kwant system representing the Kitaev chain
    """
    # Define the Pauli matrices for the Nambu space
    tau_z = np.array([[1, 0], [0, -1]])
    tau_x = np.array([[0, 1], [1, 0]])
    tau_y = np.array([[0, -1j], [1j, 0]])
    
    # Initialize the Kwant system
    syst = kwant.Builder()
    
    # Define the lattice
    lat = kwant.lattice.chain(1)
    
    # On-site Hamiltonian
    def onsite(site, disorder_val=0.0):
        # -mu σ_z term (chemical potential)
        return -mu * tau_z + disorder_val * tau_z
    
    # Add sites and on-site potentials
    if disorder is None:
        for i in range(L):
            syst[lat(i)] = onsite
    else:
        # Fixed: Create proper closures for each site by capturing the current disorder value
        for i in range(L):
            # Create a closure that properly captures the current disorder value
            disorder_val = disorder[i]
            syst[lat(i)] = lambda site, d=disorder_val: onsite(site, d)
    
    # Nearest-neighbor hopping and pairing terms
    # Hopping: -t σ_z
    # p-wave pairing: Δ (iσ_y)
    def hopping(site1, site2):
        return -t * tau_z + 1j * delta * tau_y
    
    for i in range(L-1):
        syst[lat(i), lat(i+1)] = hopping
    
    return syst

def build_kitaev_hamiltonian_kwant(L, mu, t, delta, disorder=None):
    """
    Builds the Kitaev chain Hamiltonian matrix using Kwant.
    
    Parameters:
    -----------
    L : int
        System size (number of sites)
    mu : float
        Chemical potential [eV]
    t : float
        Hopping amplitude [eV]
    delta : float
        Superconducting pairing amplitude [eV]
    disorder : np.ndarray or None
        On-site disorder potentials [eV]
    
    Returns:
    --------
    H : sparse matrix or dense ndarray
        2L×2L BdG Hamiltonian matrix
    """
    # Build the Kwant system
    syst = make_kitaev_kwant_system(L, t, mu, delta, disorder)
    
    # Finalize the system
    syst = syst.finalized()
    
    # Get the Hamiltonian as a sparse matrix
    ham_sparse = syst.hamiltonian_submatrix(sparse=True)
    
    # Convert to dense array for CuPy compatibility if needed
    return ham_sparse.toarray()

def build_kitaev_hamiltonian_direct(L, mu, t, delta, disorder=None):
    """
    Builds the Kitaev chain Hamiltonian matrix using direct construction with CuPy.
    
    Parameters:
    -----------
    L : int
        System size (number of sites)
    mu : float
        Chemical potential [eV]
    t : float
        Hopping amplitude [eV]
    delta : float
        Superconducting pairing amplitude [eV]
    disorder : cp.ndarray or None
        On-site disorder potentials [eV]
    
    Returns:
    --------
    H : cp.ndarray
        2L×2L BdG Hamiltonian matrix
    """
    # Initialize Hamiltonian as a 2L×2L matrix
    H = cp.zeros((2*L, 2*L), dtype=cp.complex64)
    
    # Apply on-site potential and disorder
    for i in range(L):
        # Diagonal terms (chemical potential)
        site_energy = -mu
        if disorder is not None:
            site_energy += disorder[i]
            
        # Electron block (upper-left)
        H[i, i] = site_energy
        
        # Hole block (lower-right) with negative sign due to particle-hole symmetry
        H[i+L, i+L] = -site_energy
    
    # Apply nearest-neighbor hopping
    for i in range(L-1):
        # Electron hopping (upper-left block)
        H[i, i+1] = -t
        H[i+1, i] = -t
        
        # Hole hopping (lower-right block)
        H[i+L, i+1+L] = t
        H[i+1+L, i+L] = t
        
        # Superconducting pairing (off-diagonal blocks)
        # p-wave pairing connects neighboring sites
        H[i, i+1+L] = delta  # upper-right block
        H[i+1, i+L] = -delta
        H[i+L, i+1] = delta  # lower-left block
        H[i+1+L, i] = -delta
    
    return H

def is_topological(mu, t, delta):
    """
    Analytical check if parameters are in the topological phase.
    For the Kitaev chain, the topological phase exists when |μ| < 2|t|
    
    Parameters:
    -----------
    mu : float
        Chemical potential [eV]
    t : float
        Hopping amplitude [eV]
    delta : float
        Superconducting pairing amplitude [eV]
        
    Returns:
    --------
    bool : True if in topological phase, False otherwise
    """
    return np.abs(mu) < 2.0 * np.abs(t) and np.abs(delta) > 0

def compute_topological_invariant(mu, t):
    """
    Compute the Z2 topological invariant for the Kitaev chain.
    The invariant is Q = sign(2|t| - |mu|)
    
    Parameters:
    -----------
    mu : float
        Chemical potential [eV]
    t : float
        Hopping amplitude [eV]
        
    Returns:
    --------
    int : Topological invariant (-1 or 1)
    """
    return np.sign(2.0 * np.abs(t) - np.abs(mu))

def has_mzm(eigenvalues, threshold=1e-6):
    """
    Check if the system has MZMs by looking for near-zero energy eigenvalues.
    
    Parameters:
    -----------
    eigenvalues : cp.ndarray or np.ndarray
        Array of energy eigenvalues
    threshold : float
        Energy threshold to identify MZMs [eV]
        
    Returns:
    --------
    bool : True if MZMs are present, False otherwise
    """
    # Look for eigenvalues close to zero
    min_energy = np.min(np.abs(eigenvalues))
    return min_energy < threshold

def localization_measure(eigenvectors, L):
    """
    Calculate localization of the lowest energy mode at the edges.
    
    Parameters:
    -----------
    eigenvectors : cp.ndarray or np.ndarray
        Matrix of eigenvectors
    L : int
        System size
        
    Returns:
    --------
    float : Measure of edge localization (higher values = more localized at edges)
    """
    # Get the lowest energy eigenvector
    zero_mode_idx = np.argmin(np.abs(np.diag(eigenvectors)))
    zero_mode = eigenvectors[:, zero_mode_idx]
    
    # Extract electron component (first L elements)
    electron_component = zero_mode[:L]
    
    # Calculate probability density
    prob = np.abs(electron_component)**2
    
    # Measure localization at edges (first and last 10% of sites)
    edge_size = max(1, int(0.1 * L))
    edge_weight = np.sum(prob[:edge_size]) + np.sum(prob[L-edge_size:])
    
    return float(edge_weight)

def generate_random_parameters(params, size):
    """
    Generate random parameter configurations for the Kitaev chain.
    
    Parameters:
    -----------
    params : KitaevChainParameters
        Parameter ranges
    size : int
        Number of configurations to generate
        
    Returns:
    --------
    dict : Dictionary containing arrays of parameters
    """
    # System size (integer values)
    L_values = np.random.randint(params.L_min, params.L_max + 1, size=size)
    
    # Generate continuous parameters
    mu_values = np.random.uniform(params.mu_min, params.mu_max, size=size)
    t_values = np.random.uniform(params.t_min, params.t_max, size=size)
    delta_values = np.random.uniform(params.delta_min, params.delta_max, size=size)
    disorder_strength = np.random.uniform(params.disorder_min, params.disorder_max, size=size)
    
    return {
        'L': L_values,
        'mu': mu_values,
        't': t_values,
        'delta': delta_values,
        'disorder_strength': disorder_strength
    }

def create_non_overlapping_sets(all_params, n_train):
    """
    Split parameters into non-overlapping train and test sets.
    
    Parameters:
    -----------
    all_params : dict
        Dictionary of parameter arrays
    n_train : int
        Number of samples for training set
        
    Returns:
    --------
    train_params, test_params : tuple of dicts
        Parameter dictionaries for train and test sets
    """
    # Create indices for train/test split
    indices = np.arange(len(all_params['L']))
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    # Create train and test parameter dictionaries
    train_params = {k: v[train_idx] for k, v in all_params.items()}
    test_params = {k: v[test_idx] for k, v in all_params.items()}
    
    return train_params, test_params

def process_dataset(params_dict, mzm_energy_threshold, label='dataset', use_kwant=True):
    """
    Process dataset from parameters, computing Hamiltonians, eigenvalues, and MZM labels.
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of parameter arrays
    mzm_energy_threshold : float
        Energy threshold for identifying MZMs [eV]
    label : str
        Dataset label (train or test)
    use_kwant : bool
        Whether to use Kwant for Hamiltonian construction
        
    Returns:
    --------
    dict : Dictionary containing complete dataset
    """
    size = len(params_dict['L'])
    print(f"Processing {size} samples for {label} set...")
    
    # Initialize storage for results
    results = {
        'L': params_dict['L'],
        'mu': params_dict['mu'],
        't': params_dict['t'],
        'delta': params_dict['delta'],
        'disorder_strength': params_dict['disorder_strength'],
        'has_mzm': np.zeros(size, dtype=np.int32),
        'topological_invariant': np.zeros(size, dtype=np.int32),
        'min_energy': np.zeros(size, dtype=np.float32),
        'edge_localization': np.zeros(size, dtype=np.float32),
        'eigenvalues': [],
        'wavefunctions': []
    }
    
    # Process each configuration
    for i in range(size):
        if i % 100 == 0:
            print(f"  Processing sample {i}/{size}...")
            
        L = int(params_dict['L'][i])
        mu = float(params_dict['mu'][i])
        t = float(params_dict['t'][i])
        delta = float(params_dict['delta'][i])
        disorder_strength = float(params_dict['disorder_strength'][i])
        
        # Compute topological invariant (theoretical prediction)
        results['topological_invariant'][i] = compute_topological_invariant(mu, t)
        
        # Generate disorder potential if needed
        disorder = None
        if disorder_strength > 0:
            disorder = np.random.uniform(-disorder_strength, disorder_strength, size=L)
        
        # Build Hamiltonian using Kwant or direct method
        if use_kwant:
            H_np = build_kitaev_hamiltonian_kwant(L, mu, t, delta, disorder)
            # Convert to CuPy array for GPU acceleration
            H = cp.array(H_np)
        else:
            # Convert disorder to CuPy if needed
            if disorder is not None:
                disorder = cp.array(disorder)
            H = build_kitaev_hamiltonian_direct(L, mu, t, delta, disorder)
        
        # Diagonalize to get eigenvalues and eigenvectors
        try:
            # Use CuPy's CUDA-accelerated eigenvalue solver
            eigenvalues, eigenvectors = cp.linalg.eigh(H)
        except cp.cuda.memory.OutOfMemoryError:
            # Fall back to CPU if GPU memory is insufficient
            print(f"  CUDA OOM for sample {i} with L={L}. Falling back to CPU...")
            if use_kwant:
                # Already have H_np
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
            else:
                H_np = cp.asnumpy(H)
                eigenvalues, eigenvectors = np.linalg.eigh(H_np)
                
            # Convert back to numpy arrays
            eigenvalues_np = eigenvalues
            eigenvectors_np = eigenvectors
        else:
            # Convert back to numpy for further processing
            eigenvalues_np = cp.asnumpy(eigenvalues)
            eigenvectors_np = cp.asnumpy(eigenvectors)
        
        # Check for MZMs
        mzm_present = has_mzm(eigenvalues_np, mzm_energy_threshold)
        edge_loc = localization_measure(eigenvectors_np, L)
        
        # Store results
        results['has_mzm'][i] = int(mzm_present)
        results['min_energy'][i] = float(np.min(np.abs(eigenvalues_np)))
        results['edge_localization'][i] = float(edge_loc)
        
        # Store eigenvalues and wavefunctions (first few only to save space)
        results['eigenvalues'].append(eigenvalues_np)
        
        # Store only the lowest energy wavefunctions
        zero_mode_idx = np.argmin(np.abs(eigenvalues_np))
        wavefunction = eigenvectors_np[:, zero_mode_idx]
        results['wavefunctions'].append(wavefunction)
    
    # Convert eigenvalues and wavefunctions to numpy arrays of objects
    results['eigenvalues'] = np.array(results['eigenvalues'], dtype=object)
    results['wavefunctions'] = np.array(results['wavefunctions'], dtype=object)
    
    return results

def append_batch_to_hdf5(data, filename, group_name, chunk_size=1000):
    """
    Append a batch of data to an existing HDF5 file.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing batch data
    filename : str
        Path to HDF5 file
    group_name : str
        Group name in HDF5 file ('train' or 'test')
    chunk_size : int
        Size of chunks for processing large datasets
    """
    with h5py.File(filename, 'a') as f:
        # Get group
        if group_name in f:
            grp = f[group_name]
        else:
            grp = f.create_group(group_name)
            
        # Get the current size of each dataset
        if 'L' in grp:
            current_size = grp['L'].shape[0]
        else:
            current_size = 0
            
            # Create datasets for scalar values
            for key in ['L', 'mu', 't', 'delta', 'disorder_strength', 'has_mzm', 
                      'min_energy', 'edge_localization', 'topological_invariant']:
                dtype = np.int32 if key in ('has_mzm', 'topological_invariant', 'L') else np.float32
                grp.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)
                
            # Create spectra subgroup
            spec_grp = grp.create_group('spectra')
            
            # Create dataset for eigenvalues (variable length)
            spec_grp.create_dataset('eigenvalues', shape=(0,), maxshape=(None,), 
                                 dtype=h5py.special_dtype(vlen=np.float64), chunks=True)
            
            # For wavefunctions, determine the maximum length across all wavefunctions
            max_wf_len = max(len(wf) for wf in data['wavefunctions'])
            spec_grp.create_dataset('wavefunctions', shape=(0, max_wf_len), maxshape=(None, max_wf_len), 
                                 dtype=np.complex64, chunks=True)
        
        # Get number of samples in this batch
        batch_size = len(data['L'])
        
        # Process in chunks to avoid memory issues
        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(batch_size, chunk_start + chunk_size)
            chunk_n = chunk_end - chunk_start
            
            # Resize datasets for scalar values
            for key in ['L', 'mu', 't', 'delta', 'disorder_strength', 'has_mzm', 
                      'min_energy', 'edge_localization', 'topological_invariant']:
                ds = grp[key]
                ds.resize((current_size + chunk_end - chunk_start,))
                ds[current_size:current_size+chunk_n] = data[key][chunk_start:chunk_end]
            
            # Handle eigenvalues (variable length)
            eig_ds = grp['spectra/eigenvalues']
            eig_ds.resize((current_size + chunk_n,))
            for i, idx in enumerate(range(chunk_start, chunk_end)):
                eig_ds[current_size + i] = data['eigenvalues'][idx]
            
            # Handle wavefunctions
            wf_ds = grp['spectra/wavefunctions']
            max_wf_len = wf_ds.shape[1]
            wf_ds.resize((current_size + chunk_n, max_wf_len))
            
            for i, idx in enumerate(range(chunk_start, chunk_end)):
                wf = data['wavefunctions'][idx]
                if len(wf) < max_wf_len:
                    # Pad if necessary
                    padded = np.zeros(max_wf_len, dtype=np.complex64)
                    padded[:len(wf)] = wf
                    wf_ds[current_size + i] = padded
                else:
                    wf_ds[current_size + i] = wf[:max_wf_len]
            
            # Update current_size for next chunk
            current_size += chunk_n

def save_to_hdf5(train_data, test_data, filename, mzm_energy_threshold, chunk_size=5000):
    """
    Save the dataset to HDF5 format in streaming chunks to handle very large datasets.
    """
    with h5py.File(filename, 'w') as f:
        # metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['total_samples'] = N_TOTAL
        f.attrs['description'] = 'Kitaev chain dataset for ML-based MZM detection'
        f.attrs['mzm_energy_threshold'] = mzm_energy_threshold
        f.attrs['implementation'] = 'Kwant + CuPy'

        for group_name, data in [('train', train_data), ('test', test_data)]:
            grp = f.create_group(group_name)
            size = len(data['L'])

            # create resizable datasets for scalar fields
            dsets = {}
            for key in ['L','mu','t','delta','disorder_strength','has_mzm',
                      'min_energy','edge_localization','topological_invariant']:
                dtype = 'int32' if key in ('has_mzm','topological_invariant') else 'float32'
                dsets[key] = grp.create_dataset(
                    key, shape=(0,), maxshape=(None,), dtype=dtype, chunks=True)

            # create spectra subgroup
            spec = grp.create_group('spectra')
            # eigenvalues: variable-length floats
            dt_real = h5py.special_dtype(vlen=float)
            eig_ds = spec.create_dataset(
                'eigenvalues', shape=(0,), maxshape=(None,), dtype=dt_real, chunks=True)
            # wavefunctions: pad to max_len across entire set
            max_len = max(len(wf) for wf in data['wavefunctions'])
            wf_ds = spec.create_dataset(
                'wavefunctions', shape=(0, max_len),
                maxshape=(None, max_len), dtype=np.complex64, chunks=True)

            # write in chunks
            for start in range(0, size, chunk_size):
                end = min(size, start + chunk_size)
                n = end - start

                # resize all datasets
                for ds in dsets.values():
                    ds.resize((ds.shape[0] + n,))
                eig_ds.resize((eig_ds.shape[0] + n,))
                wf_ds.resize((wf_ds.shape[0] + n, max_len))

                # write chunk
                for key, ds in dsets.items():
                    if key in data:  # Check if key exists in data
                        arr = data[key][start:end]
                        ds[-n:] = arr

                for i, idx in enumerate(range(start, end)):
                    eig_ds[-n + i] = data['eigenvalues'][idx]
                    wf = data['wavefunctions'][idx]
                    if len(wf) < max_len:
                        padded = np.zeros(max_len, dtype=np.complex64)
                        padded[:len(wf)] = wf
                        wf_ds[-n + i, :] = padded
                    else:
                        wf_ds[-n + i, :] = wf.astype(np.complex64)

def analyze_dataset(data):
    """
    Analyze dataset to report statistics.
    
    Parameters:
    -----------
    data : dict
        Dataset dictionary
    
    Returns:
    --------
    dict : Dictionary of statistics
    """
    stats = {
        'size': len(data['L']),
        'mzm_count': int(np.sum(data['has_mzm'])),
        'mzm_percentage': float(100 * np.sum(data['has_mzm']) / len(data['has_mzm'])),
        'avg_min_energy': float(np.mean(data['min_energy'])),
        'avg_localization': float(np.mean(data['edge_localization'])),
        'topological_matches': float(100 * np.sum(data['has_mzm'] == (data['topological_invariant'] > 0)) / len(data['has_mzm']))
    }
    return stats

def plot_sample_spectrum(data, sample_idx, output_dir):
    """
    Plot the energy spectrum for a sample.
    
    Parameters:
    -----------
    data : dict
        Dataset dictionary
    sample_idx : int
        Index of the sample to plot
    output_dir : str
        Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get eigenvalues and parameters
        eigenvalues = data['eigenvalues'][sample_idx]
        L = data['L'][sample_idx]
        mu = data['mu'][sample_idx]
        t = data['t'][sample_idx]
        delta = data['delta'][sample_idx]
        has_mzm = data['has_mzm'][sample_idx]
        
        # Plot the low-energy spectrum
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(eigenvalues)), eigenvalues, 'o-', markersize=5)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title(f"Sample {sample_idx}: L={L}, μ={mu:.2f}, t={t:.2f}, Δ={delta:.2f}")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Energy (eV)")
        plt.grid(True, alpha=0.3)
        
        if has_mzm:
            plt.text(0.05, 0.95, "MZM Present", transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='green', alpha=0.5))
        else:
            plt.text(0.05, 0.95, "No MZM", transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='red', alpha=0.5))
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f"spectrum_sample_{sample_idx}.png"), dpi=300)
        plt.close()
        
        # Plot the wavefunction density for the lowest mode
        plt.figure(figsize=(10, 6))
        wavefunction = data['wavefunctions'][sample_idx]
        L_sites = int(len(wavefunction) / 2)  # Half for electrons, half for holes
        
        # Get electron and hole components
        electron_component = wavefunction[:L_sites]
        hole_component = wavefunction[L_sites:]
        
        plt.plot(np.arange(L_sites), np.abs(electron_component)**2, 'b-', label='Electron')
        plt.plot(np.arange(L_sites), np.abs(hole_component)**2, 'r-', label='Hole')
        
        plt.title(f"Sample {sample_idx}: Zero Mode Wavefunction")
        plt.xlabel("Site index")
        plt.ylabel("Probability density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(plots_dir, f"wavefunction_sample_{sample_idx}.png"), dpi=300)
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for plotting")

def configure_gpu_memory():
    """
    Configure GPU memory settings for the RTX A5000.
    This helps prevent out-of-memory errors by limiting memory growth.
    """
    try:
        # Try to get device info and print memory stats
        device = cp.cuda.Device()
        print(f"CUDA Device: {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")
        print(f"Total Memory: {device.mem_info[1]/1024**3:.2f} GB")
        
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(fraction=0.85)  # Use 85% of available GPU memory
        print(f"Memory pool limit set to: {mempool.get_limit()/1024**3:.2f} GB")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        print("Continuing with default memory settings")


def main():
    """
    Main execution function for the Kitaev chain dataset generator.
    """
    start_time = time.time()
    
    # Configure GPU memory
    configure_gpu_memory()
    
    # Initialize parameter object
    params = KitaevChainParameters()
    
    # Define output paths
    output_dir = "kitaev_chain_dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"kitaev_chain_dataset_{N_TRAIN}_{N_TEST}.h5")
    
    print(f"Generating dataset with {N_TRAIN} training samples and {N_TEST} test samples...")
    
    # Generate all parameter configurations
    print("Generating random parameter configurations...")
    all_params = generate_random_parameters(params, N_TOTAL)
    
    # Split into train and test sets (ensure no overlap in parameter space)
    print("Creating non-overlapping train and test sets...")
    train_params, test_params = create_non_overlapping_sets(all_params, N_TRAIN)
    
    # Process training dataset
    print("\nProcessing training dataset...")
    train_data = process_dataset(train_params, params.mzm_energy_threshold, label='train', use_kwant=True)
    
    # Process test dataset
    print("\nProcessing test dataset...")
    test_data = process_dataset(test_params, params.mzm_energy_threshold, label='test', use_kwant=True)
    
    # Save datasets to HDF5 format
    print("\nSaving datasets to HDF5 format...")
    save_to_hdf5(train_data, test_data, output_file, params.mzm_energy_threshold)
    
    # Analyze datasets
    print("\nAnalyzing dataset statistics...")
    train_stats = analyze_dataset(train_data)
    test_stats = analyze_dataset(test_data)
    
    print("\nTraining set statistics:")
    print(f"  Samples: {train_stats['size']}")
    print(f"  MZM samples: {train_stats['mzm_count']} ({train_stats['mzm_percentage']:.2f}%)")
    print(f"  Average min energy: {train_stats['avg_min_energy']:.6f} eV")
    print(f"  Average edge localization: {train_stats['avg_localization']:.6f}")
    print(f"  Theory-simulation match: {train_stats['topological_matches']:.2f}%")
    
    print("\nTest set statistics:")
    print(f"  Samples: {test_stats['size']}")
    print(f"  MZM samples: {test_stats['mzm_count']} ({test_stats['mzm_percentage']:.2f}%)")
    print(f"  Average min energy: {test_stats['avg_min_energy']:.6f} eV")
    print(f"  Average edge localization: {test_stats['avg_localization']:.6f}")
    print(f"  Theory-simulation match: {test_stats['topological_matches']:.2f}%")
    
    # Plot example spectra
    print("\nGenerating example plots...")
    # Plot one MZM and one non-MZM sample from training set
    mzm_idx = np.where(train_data['has_mzm'] == 1)[0][0]
    non_mzm_idx = np.where(train_data['has_mzm'] == 0)[0][0]
    plot_sample_spectrum(train_data, mzm_idx, output_dir)
    plot_sample_spectrum(train_data, non_mzm_idx, output_dir)
    
    # Calculate and report runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/60:.2f} minutes")
    print(f"Dataset successfully saved to: {output_file}")
    
    # Save metadata to a text file
    metadata_file = os.path.join(output_dir, "dataset_metadata.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"Kitaev Chain Dataset\n")
        f.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training samples: {train_stats['size']}\n")
        f.write(f"Test samples: {test_stats['size']}\n")
        f.write(f"MZM energy threshold: {params.mzm_energy_threshold}\n")
        f.write(f"Parameter ranges:\n")
        f.write(f"  L: [{params.L_min}, {params.L_max}]\n")
        f.write(f"  mu: [{params.mu_min}, {params.mu_max}] eV\n")  # Changed μ to mu
        f.write(f"  t: [{params.t_min}, {params.t_max}] eV\n")
        f.write(f"  Delta: [{params.delta_min}, {params.delta_max}] eV\n")  # Changed Δ to Delta
        f.write(f"  Disorder: [{params.disorder_min}, {params.disorder_max}] eV\n")
        f.write(f"\nTraining set statistics:\n")
        f.write(f"  MZM samples: {train_stats['mzm_count']} ({train_stats['mzm_percentage']:.2f}%)\n")
        f.write(f"  Average min energy: {train_stats['avg_min_energy']:.6f} eV\n")
        f.write(f"  Theory-simulation match: {train_stats['topological_matches']:.2f}%\n")
        f.write(f"\nTest set statistics:\n")
        f.write(f"  MZM samples: {test_stats['mzm_count']} ({test_stats['mzm_percentage']:.2f}%)\n")
        f.write(f"  Average min energy: {test_stats['avg_min_energy']:.6f} eV\n")
        f.write(f"  Theory-simulation match: {test_stats['topological_matches']:.2f}%\n")
        f.write(f"\nTotal runtime: {total_time/60:.2f} minutes\n")
        f.write(f"Dataset saved to: {output_file}\n")
        f.write(f"Plots saved to: {os.path.join(output_dir, 'plots')}\n")
    print(f"Metadata saved to: {metadata_file}")
    # Print completion message
    print("Dataset generation and analysis completed successfully.")


if __name__ == "__main__":
    main()