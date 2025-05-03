import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pywt
from matplotlib.colors import Normalize
from PIL import Image
import argparse
import pyttsx3

# --- Raw Signal Model definition (from train_raw_signal_model.py) ---
class RawSignalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(3,1,5), padding=(1,0,2)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,1,3), padding=(1,0,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Model definition (copied from streamlit_asl_infer.py) ---
class SimpleJetsonCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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

# --- Load continuous test data from xlsx ---
def load_continuous_xlsx(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)
    sheets = xls.sheet_names
    if len(sheets) != 5:
        raise ValueError(f"Expected 5 sheets (one per sensor), got {len(sheets)})")
    sensor_arrays = []
    for sheet in sheets:
        df = xls.parse(sheet, header=None)
        if df.shape[1] < 8:
            raise ValueError(f"Sheet '{sheet}' has only {df.shape[1]} columns, expected at least 8 (need columns 3-8)")
        feature_df = df.iloc[:, 2:8]  # columns 3-8
        sensor_arrays.append(feature_df.values)
    min_timesteps = min(arr.shape[0] for arr in sensor_arrays)
    arr = np.stack([arr[:min_timesteps, :] for arr in sensor_arrays], axis=0)  # (5, T, 6)
    return arr  # shape: (5, T, 6)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream test data through ASL model in sliding window fashion.")
    parser.add_argument('--file', type=str, default="test_data.xlsx", help="Path to test .xlsx file")
    parser.add_argument('--model', type=str, default="best_scalogram_cnn.pth", help="Path to model .pth file")
    parser.add_argument('--window', type=int, default=91, help="Window size (samples)")
    parser.add_argument('--stride', type=int, default=91, help="Stride for sliding window")
    parser.add_argument('--mode', type=str, default="scalogram", choices=["scalogram", "rawsignal"], help="Model type: scalogram or rawsignal")
    args = parser.parse_args()

    # Load data
    if args.file.lower().endswith('.csv'):
        data = load_continuous_csv(args.file)
    else:
        data = load_continuous_xlsx(args.file)
    T = data.shape[1]
    file_type = "CSV" if args.file.lower().endswith('.csv') else "Excel"
    print(f"Loaded {file_type} data shape: {data.shape}")

    num_classes = 27
    class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','Rest','S','T','U','V','W','X','Y','Z']
    if args.mode == "scalogram":
        model = SimpleJetsonCNN(num_classes=num_classes)
    else:
        model = RawSignalCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    import matplotlib.pyplot as plt
    import time

    # Prepare for visualization
    timeline_preds = []
    timeline_starts = []
    timeline_probs = []
    plt.ion()
    fig, (ax_img, ax_bar, ax_timeline) = plt.subplots(1, 3, figsize=(18, 5))
    bar_container = None
    timeline_line = None

    for start in range(0, T - args.window + 1, args.stride):
        window = data[:, start:start+args.window, :]  # (5, window, 6)
        if args.mode == "scalogram":
            grid_img = make_scalogram_grid(window)
            img_pil = Image.fromarray((grid_img*255).astype(np.uint8))
            img_pil = img_pil.resize((128, 128))
            img_arr = np.array(img_pil).astype(np.float32) / 255.0
            img_arr = (img_arr - 0.5) / 0.5
            img_arr = np.transpose(img_arr, (2, 0, 1))  # (3, H, W)
            img_tensor = torch.tensor(img_arr).unsqueeze(0)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
            # --- Visualization ---
            ax_img.clear()
            ax_img.imshow(np.transpose(img_arr, (1, 2, 0)) * 0.5 + 0.5)
            ax_img.set_title(f'Scalogram Grid (Window {start}-{start+args.window-1})')
            ax_img.axis('off')
        else:
            # Raw signal model expects (6, 5, 1, window)
            x = window.astype(np.float32)  # (5, window, 6)
            x = np.transpose(x, (2, 0, 1))  # (6, 5, window)
            x = np.expand_dims(x, axis=2)  # (6, 5, 1, window)
            x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, 6, 5, 1, window)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
            # --- Visualization ---
            ax_img.clear()
            # Show a simple visualization: mean of all features as an image
            mean_img = x.mean(axis=0).squeeze()  # (5, window)
            ax_img.imshow(mean_img, aspect='auto', cmap='viridis')
            ax_img.set_title(f'Raw Signal (Window {start}-{start+args.window-1})')
            ax_img.axis('off')

        ax_bar.clear()
        bar_container = ax_bar.bar(class_names, probs, color='skyblue')
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title('Class Probabilities')
        ax_bar.set_ylabel('Probability')
        ax_bar.set_xticks(range(len(class_names)))
        ax_bar.set_xticklabels(class_names, rotation=90)
        ax_bar.axhline(0.6, color='red', linestyle='--', linewidth=0.8)
        ax_bar.bar(class_names[pred_idx], probs[pred_idx], color='orange')
        ax_bar.text(pred_idx, probs[pred_idx]+0.03, f'{class_names[pred_idx]}\n{probs[pred_idx]:.2f}', ha='center', color='black')

        timeline_preds.append(pred_idx)
        timeline_starts.append(start)
        timeline_probs.append(probs[pred_idx])
        ax_timeline.clear()
        ax_timeline.plot(timeline_starts, timeline_preds, drawstyle='steps-post', color='blue')
        ax_timeline.set_yticks(range(len(class_names)))
        ax_timeline.set_yticklabels(class_names)
        ax_timeline.set_xlabel('Start Index (Time)')
        ax_timeline.set_ylabel('Predicted Class')
        ax_timeline.set_title('Prediction Timeline')
        ax_timeline.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax_timeline.set_xlim(0, T)
        ax_timeline.set_ylim(-1, len(class_names))

        plt.tight_layout()
        plt.pause(0.2)  # Pause to simulate streaming

        # --- Audio output for predicted class (offline, using pyttsx3) ---
        if start == 0:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
        if probs[pred_idx] > 0.6:
            tts_engine.say(class_names[pred_idx])
            tts_engine.runAndWait()
    plt.ioff()
    plt.show()
