import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Path to dataset directory
dataset_dir = "kitaev_chain_dataset"

# Collect all .h5 files
h5_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.h5')]

# Select one random file and group
h5_path = random.choice(h5_files)
group_name = random.choice(['train', 'test'])

# Open file and select random sample index
with h5py.File(h5_path, 'r') as f:
    group = f[group_name]
    num_samples = group['min_energy'].shape[0]
    idx = random.randrange(num_samples)
    
    # Retrieve full eigenvalue vector
    eig = group['spectra']['eigenvalues'][idx]
    
    # Print file, group, and index
    print(f"File: {h5_path}")
    print(f"Group: {group_name}")
    print(f"Random Sample Index: {idx}")
    print("Eigenvalues:")
    print(eig)
    
    # Plot eigenvalues
    plt.figure(figsize=(6, 4))
    plt.plot(eig, marker='o')
    plt.title("Full Eigenvalue Spectrum for One Random Sample")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig("eigenvalue_spectrum.png")
