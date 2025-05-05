#!/usr/bin/env python3
"""
CSV Scalogram Visualization using the exact same methods as stream_real_data.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pywt
from matplotlib.colors import Normalize
from PIL import Image
import argparse
import os
import sys
import matplotlib.pyplot as plt
import time

# --- Utility: Generate RGB scalogram for 3-axis data ---
def rgb_scalogram_3axis(sig_x, sig_y, sig_z, scales=np.arange(1,31), wavelet='morl'):
    rgb = np.zeros((len(scales), sig_x.shape[0], 3))
    for i, sig in enumerate([sig_x, sig_y, sig_z]):
        cwtmatr, _ = pywt.cwt(sig, scales, wavelet)
        norm = Normalize()(np.abs(cwtmatr))
        rgb[..., i] = norm
    rgb = rgb / np.max(rgb)
    return rgb

# --- Generate 2x5 grid scalogram image from sample ---
def make_scalogram_grid(sample):
    SCALES = np.arange(1, 31)
    WAVELET = 'morl'
    row_gyro = []
    row_acc = []
    for sensor_idx in range(5):
        sensor_data = sample[sensor_idx]  # (window, 6)
        # Gyro (0,1,2)
        rgb_gyro = rgb_scalogram_3axis(sensor_data[:,0], sensor_data[:,1], sensor_data[:,2], SCALES, WAVELET)
        row_gyro.append(rgb_gyro)
        # Acc (3,4,5)
        rgb_acc = rgb_scalogram_3axis(sensor_data[:,3], sensor_data[:,4], sensor_data[:,5], SCALES, WAVELET)
        row_acc.append(rgb_acc)
    grid_row1 = np.concatenate(row_gyro, axis=1)
    grid_row2 = np.concatenate(row_acc, axis=1)
    grid_img = np.concatenate([grid_row1, grid_row2], axis=0)  # (2*scales, 5*window, 3)
    return grid_img

# --- Load continuous test data from CSV ---
def load_continuous_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 8:
        raise ValueError(f"CSV has only {df.shape[1]} columns, expected at least 8 (timestamp, sensor, 6 features)")
    sensor_arrays = []
    for sensor_num in range(5):
        sensor_df = df[df.iloc[:, 1] == sensor_num]
        features = sensor_df.iloc[:, 2:8].values  # shape (T, 6)
        sensor_arrays.append(features)
    min_timesteps = min(arr.shape[0] for arr in sensor_arrays)
    arr = np.stack([arr[:min_timesteps, :] for arr in sensor_arrays], axis=0)  # (5, T, 6)
    return arr  # shape: (5, T, 6)

def main():
    parser = argparse.ArgumentParser(description="CSV Scalogram Visualization")
    parser.add_argument('csv_file', type=str, help="Path to CSV file")
    parser.add_argument('--window', type=int, default=91, help="Window size (samples)")
    parser.add_argument('--stride', type=int, default=45, help="Stride between windows")
    parser.add_argument('--save', action='store_true', help="Save scalogram images")
    parser.add_argument('--output-dir', type=str, default="scalograms", help="Directory to save scalogram images")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)
    
    # Create output directory if saving images
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.csv_file}...")
    try:
        data = load_continuous_csv(args.csv_file)
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Get total time steps
    T = data.shape[1]
    print(f"Total time steps: {T}")
    print(f"Window size: {args.window}")
    print(f"Stride: {args.stride}")
    
    # Setup visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Scalogram Visualization")
    ax.axis('off')
    
    # Process data in windows
    window_count = (T - args.window) // args.stride + 1
    print(f"Total windows: {window_count}")
    
    for i, start in enumerate(range(0, T - args.window + 1, args.stride)):
        # Get window data
        window = data[:, start:start+args.window, :]  # (5, window, 6)
        
        # Generate scalogram
        grid_img = make_scalogram_grid(window)
        
        # Display scalogram
        ax.clear()
        ax.imshow(grid_img)
        ax.set_title(f"Window {i+1}/{window_count} (Frames {start}-{start+args.window-1})")
        ax.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # Save image if requested
        if args.save:
            img_pil = Image.fromarray((grid_img*255).astype(np.uint8))
            img_path = os.path.join(args.output_dir, f"scalogram_{i:04d}.png")
            img_pil.save(img_path)
            print(f"Saved {img_path}")
    
    # Keep the plot open
    plt.ioff()
    print("Press any key to close...")
    plt.show()

if __name__ == "__main__":
    main()
