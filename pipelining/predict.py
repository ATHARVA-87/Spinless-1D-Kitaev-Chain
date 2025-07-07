#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction script for Kitaev chain 1D BdG model.

This script loads a pre-trained BdGPredictor model and generates predictions
for the edge localization and BdG eigenstates based on 5 input parameters:
- L: Chain length
- mu: Chemical potential (eV)
- t: Hopping parameter (eV)
- delta: Superconducting gap (eV)
- disorder_strength: Disorder strength (eV)

The model predicts:
- Edge localization (majorana zero mode presence indicator)
- 2*L BdG eigenstates

Usage:
    Just run the script directly in VSCode or terminal: python predict.py

Author: [Your Name]
Date: May 09, 2025
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import BdGPredictor

def load_model(model_path, input_dim=5, hidden_dim1=256, hidden_dim2=512, output_dim=181):
    """
    Load a pretrained BdGPredictor model.
    
    The model has a fixed output dimension of 181, regardless of the chain length L:
    - First element (index 0): Edge localization parameter
    - Next 2*L elements: BdG eigenstates
    - Remaining elements (181 - 1 - 2*L): Masked values (should be close to 0)
    
    Args:
        model_path (str): Path to the saved model file
        input_dim (int): Input dimension of the model (default: 5)
        hidden_dim1 (int): First hidden layer dimension (default: 256)
        hidden_dim2 (int): Second hidden layer dimension (default: 512)
        output_dim (int): Output dimension of the model (default: 181)
        
    Returns:
        model: Loaded BdGPredictor model
        device: PyTorch device (cuda/cpu)
    """
    # Initialize model with the same architecture as used during training
    model = BdGPredictor(input_dim=input_dim, hidden_dim1=hidden_dim1, 
                        hidden_dim2=hidden_dim2, output_dim=output_dim)
    
    # Log model architecture
    print(f"Model architecture:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Hidden dimensions: {hidden_dim1}, {hidden_dim2}")
    print(f"  - Output dimension: {output_dim}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load the state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fix: Handle the case where the saved file contains a dictionary with multiple state dicts
    try:
        if device.type == 'cuda':
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=device)
        
        # Check if the loaded object is a dictionary with 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Found checkpoint dictionary with model_state_dict. Extracting model state...")
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume it's already a state dict
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return model, device

def predict_bdg_states(model, input_params, device, L):
    """
    Generate predictions for edge localization and BdG eigenstates.
    
    This function handles the fixed output size of 181 elements where:
    - First element (index 0) is the edge localization parameter
    - Next 2*L elements are the BdG eigenstates
    - Remaining elements (181 - 1 - 2*L) should be masked (values near 0)
    
    Args:
        model: BdGPredictor model
        input_params (list): List of 5 input parameters [L, mu, t, delta, disorder_strength]
        device: PyTorch device (cuda/cpu)
        L (int): Chain length parameter
        
    Returns:
        edge_localization (float): Predicted edge localization
        bdg_eigenstates (list): List of predicted BdG eigenstates (2*L elements)
    """
    # Convert input parameters to tensor
    input_tensor = torch.tensor([input_params], dtype=torch.float32).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert output tensor to numpy array
    output = output.cpu().numpy()[0]
    
    # Total output size should be 181 (fixed regardless of L)
    total_output_size = 181
    
    # Verify output shape
    if len(output) != total_output_size:
        print(f"WARNING: Output size {len(output)} does not match expected size {total_output_size}")
    
    # Extract edge localization (first element)
    edge_localization = float(output[0])
    print("HERE" ,output)   
    # Extract BdG eigenstates (next 2*L elements)
    num_bdg_states = min(2 * int(L), total_output_size - 1)  # Safety check
    bdg_eigenstates = output[1:1+num_bdg_states].copy()  # Use copy to avoid reference issues
    
    # Check the mask region (should be near zero)
    mask_start = 1 + num_bdg_states
    mask_end = min(total_output_size, len(output))
    
    if mask_start < mask_end:
        mask = output[mask_start:mask_end]
        mask_mean = np.mean(np.abs(mask))
        mask_max = np.max(np.abs(mask))
        print(f"Mask statistics (should be near zero):")
        print(f"  - Mean absolute value: {mask_mean:.6f}")
        print(f"  - Max absolute value: {mask_max:.6f}")
        print(f"  - Mask size: {len(mask)} elements")
    else:
        print("No mask region (all output used for edge localization and eigenstates)")
    
    print(f"Edge localization: {edge_localization:.6f}")
    print(f"Number of BdG eigenstates: {num_bdg_states} (from {L} chain sites)")
    
    return edge_localization, bdg_eigenstates

def plot_bdg_spectrum(bdg_eigenstates, L, edge_localization):
    """
    Plot the BdG eigenstate spectrum.
    
    Args:
        bdg_eigenstates (list): List of BdG eigenvalues
        L (int): Chain length
        edge_localization (float): Edge localization value
    """
    plt.figure(figsize=(12, 8))
    
    # Create main plot - BdG spectrum
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(bdg_eigenstates)), bdg_eigenstates, c='blue', marker='o', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title(f'BdG Eigenstate Spectrum (L={L})')
    plt.xlabel('Eigenstate Index')
    plt.ylabel('Energy (eV)')
    
    # Create histogram of eigenvalues
    plt.subplot(1, 2, 2)
    plt.hist(bdg_eigenstates, bins=30, color='green', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('Distribution of BdG Eigenvalues')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Count')
    
    # Add MZM status annotation
    has_mzm = edge_localization > 0.5
    mzm_status = "Predicted" if has_mzm else "Not predicted"
    min_energy = min(abs(e) for e in bdg_eigenstates)
    
    plt.figtext(0.5, 0.01, 
                f"Edge Localization: {edge_localization:.4f} | MZM: {mzm_status} | Min Energy Gap: {min_energy:.6f} eV",
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text at the bottom
    
    # Save the plot
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bdg_spectrum.png'), dpi=300)
    print(f"Spectrum plot saved to {os.path.join(output_dir, 'bdg_spectrum.png')}")
    
    # Show the plot if running in interactive mode
    plt.show()

def main():
    """Main function to run the BdG prediction for Kitaev chain 1D model.
    
    This script takes 5 input parameters for a Kitaev chain and predicts:
    1. Edge localization (indicator of Majorana Zero Mode presence)
    2. 2*L BdG eigenstates
    
    The model has a fixed output size of 181 elements, where the first element
    is the edge localization, the next 2*L elements are BdG eigenstates, and
    the remaining elements are masked (values close to 0).
    """
    
    # Set hard-coded input parameters (modify these as needed)
    MODEL_PATH = "/home/levi/anaconda3/envs/mzm/project_mzm/trained_model/run_20250509_011133/best_model.pth"  # Path to your trained model
    OUTPUT_FILE = "/home/levi/anaconda3/envs/mzm/project_mzm/pipelining/prediction_outputs/prediction_results.json"  # Output file to save results
    FIXED_OUTPUT_DIM = 181  # Fixed output dimension of the model
    
    # Kitaev chain parameters
    L = 61.0  # Chain length
    mu = -2.1203324794769287  # Chemical potential (eV)
    t = 1.9805852174758911  # Hopping parameter (eV)
    delta = 0.60959792137146  # Superconducting gap (eV)
    disorder_strength = 0.1387556493282318  # Disorder strength (eV)
    
    # Print dataset information for reference
    print("=" * 50)
    print("Kitaev Chain 1D BdG Predictor")
    print("=" * 50)
    print("\nDataset Information:")
    print("  Parameter ranges:")
    print("    L: [30, 100]")
    print("    mu: [-4.0, 4.0] eV")
    print("    t: [0.5, 2.0] eV")
    print("    Delta: [0.2, 1.5] eV")
    print("    Disorder: [0.0, 0.5] eV")
    print("  MZM energy threshold: 1e-06 eV")
    print("  Output dimension: 181 (1 edge loc. + 2*L BdG eigenstates + mask)")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print("Please adjust the MODEL_PATH variable in the script.")
        return
    
    # Check that L is within expected range
    if L < 30 or L > 100:
        print(f"WARNING: Chain length L={L} is outside the dataset range [30, 100]")
        print("Model predictions may be less reliable.")
    
    # Verify that 2*L + 1 <= FIXED_OUTPUT_DIM
    if 2*L + 1 > FIXED_OUTPUT_DIM:
        print(f"ERROR: Chain length L={L} is too large for output dimension {FIXED_OUTPUT_DIM}")
        print(f"Maximum supported L is {(FIXED_OUTPUT_DIM - 1) // 2}")
        return
    
    # Log device info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the pre-trained model
    print(f"\nLoading model from {MODEL_PATH}...")
    model, device = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # Prepare input parameters - must match the expected model input order
    input_params = [L, mu, t, delta, disorder_strength]
    print(f"\nInput parameters:")
    print(f"  L: {L}")
    print(f"  mu: {mu} eV")
    print(f"  t: {t} eV")
    print(f"  delta: {delta} eV")
    print(f"  disorder_strength: {disorder_strength} eV")
    
    # Generate predictions
    print(f"\nGenerating predictions for Kitaev chain with L = {L}...")
    edge_localization, bdg_eigenstates = predict_bdg_states(model, input_params, device, L)
    
    # Print results
    print("\nPrediction Results:")
    print("-" * 30)
    print(f"Edge Localization: {edge_localization:.6f}")
    
    # Determine if there's likely a Majorana Zero Mode (MZM)
    has_mzm = "Yes" if edge_localization < 0.5 else "No"
    print(f"Predicted to have MZM: {has_mzm}")
    
    # Calculate minimum energy gap
    min_energy = min(abs(e) for e in bdg_eigenstates)
    print(f"Minimum energy gap: {min_energy:.6f} eV")
    
    # Find the number of near-zero eigenvalues (potential MZMs)
    zero_threshold = 1e-3  # Define a threshold for "near zero"
    near_zero_states = sum(1 for e in bdg_eigenstates if abs(e) < zero_threshold)
    print(f"Number of near-zero energy states (|E| < {zero_threshold}): {near_zero_states}")
    
    # Print BdG eigenstates (first few only for readability)
    print(f"\nBdG Eigenstates ({len(bdg_eigenstates)} states in total):")
    show_states = min(10, len(bdg_eigenstates))  # Show at most 10 states
    for i, eigenvalue in enumerate(sorted(bdg_eigenstates)[:show_states]):
        print(f"  State {i+1}: {eigenvalue:.6f} eV")
    if len(bdg_eigenstates) > show_states:
        print(f"  ... {len(bdg_eigenstates)-show_states} more states (omitted for brevity) ...")
    
    # Plot the BdG spectrum
    plot_bdg_spectrum(bdg_eigenstates, L, edge_localization)
    
    # Save results to file
    results = {
        'input_params': {
            'L': float(L),
            'mu': float(mu),
            't': float(t),
            'delta': float(delta),
            'disorder_strength': float(disorder_strength)
        },
        'edge_localization': float(edge_localization),
        'has_mzm': has_mzm,
        'min_energy': float(min_energy),
        'near_zero_states': int(near_zero_states),
        'bdg_eigenstates': [float(val) for val in bdg_eigenstates]
    }
    
    # Save as JSON
    import json
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()