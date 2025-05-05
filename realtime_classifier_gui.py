import numpy as np
import torch
import torch.nn as nn
import subprocess
import threading
import time
import argparse
import queue
import signal
import sys
import os
from collections import deque
import tkinter as tk
from tkinter import ttk, font, Canvas
import pywt
from PIL import Image, ImageTk
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from io import BytesIO

# --- Scalogram Model definition ---
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
    rgb = rgb / np.max(rgb) if np.max(rgb) > 0 else rgb
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

# --- END scalogram utilities ---

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

class ASLClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("ASL Classifier")
        master.geometry("800x600")
        master.configure(bg='#2E2E2E')
        
        # Set up fonts
        self.title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.letter_font = font.Font(family="Helvetica", size=72, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=14)
        self.status_font = font.Font(family="Helvetica", size=12)
        
        # Create frames
        self.top_frame = tk.Frame(master, bg='#2E2E2E')
        self.top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.letter_frame = tk.Frame(master, bg='#2E2E2E')
        self.letter_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.confidence_frame = tk.Frame(master, bg='#2E2E2E')
        self.confidence_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.history_frame = tk.Frame(master, bg='#2E2E2E')
        self.history_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.viz_frame = tk.Frame(master, bg='#2E2E2E')
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.viz_label = tk.Label(self.viz_frame, text="Scalogram Visualization:", font=self.label_font, fg='white', bg='#2E2E2E')
        self.viz_label.pack(anchor='w')
        
        self.viz_canvas = Canvas(self.viz_frame, width=600, height=200, bg='#1E1E1E', highlightthickness=0)
        self.viz_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.status_frame = tk.Frame(master, bg='#2E2E2E')
        self.status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Title
        self.title_label = tk.Label(
            self.top_frame, 
            text="ASL Classifier", 
            font=self.title_font,
            fg='white',
            bg='#2E2E2E'
        )
        self.title_label.pack(pady=10)
        
        # Current prediction (large letter)
        self.letter_label = tk.Label(
            self.letter_frame, 
            text="?", 
            font=self.letter_font,
            fg='white',
            bg='#2E2E2E'
        )
        self.letter_label.pack(expand=True)
        
        # Confidence bar
        self.confidence_label = tk.Label(
            self.confidence_frame, 
            text="Confidence:", 
            font=self.label_font,
            fg='white',
            bg='#2E2E2E',
            anchor='w'
        )
        self.confidence_label.pack(fill=tk.X)
        
        self.confidence_bar = ttk.Progressbar(
            self.confidence_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate'
        )
        self.confidence_bar.pack(fill=tk.X, pady=5)
        
        self.confidence_value = tk.Label(
            self.confidence_frame, 
            text="0%", 
            font=self.label_font,
            fg='white',
            bg='#2E2E2E'
        )
        self.confidence_value.pack(pady=5)
        
        # Recent predictions
        self.history_label = tk.Label(
            self.history_frame, 
            text="Recent Predictions:", 
            font=self.label_font,
            fg='white',
            bg='#2E2E2E',
            anchor='w'
        )
        self.history_label.pack(fill=tk.X)
        
        self.history_text = tk.Text(
            self.history_frame, 
            height=3, 
            font=self.status_font,
            fg='white',
            bg='#3E3E3E',
            wrap=tk.WORD
        )
        self.history_text.pack(fill=tk.X, pady=5)
        self.history_text.insert(tk.END, "Waiting for predictions...")
        self.history_text.config(state=tk.DISABLED)
        
        # Status
        self.status_label = tk.Label(
            self.status_frame, 
            text="Status: Initializing...", 
            font=self.status_font,
            fg='yellow',
            bg='#2E2E2E',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Initialize history
        self.prediction_history = []
        
        # Set up style for progress bar
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("TProgressbar", thickness=30, background='#4CAF50')
        
        # Set up periodic UI update
        self.master.after(100, self.check_queue)

    def update_prediction(self, letter, confidence, scalogram_img=None):
        """Update the UI with a new prediction"""
        # Update letter
        self.letter_label.config(text=letter)
        
        # Update confidence bar
        confidence_pct = int(confidence * 100)
        self.confidence_bar['value'] = confidence_pct
        self.confidence_value.config(text=f"{confidence_pct}%")
        
        # Change color based on confidence
        if confidence >= 0.7:
            self.style.configure("TProgressbar", background='#4CAF50')  # Green
            self.letter_label.config(fg='#4CAF50')
        elif confidence >= 0.4:
            self.style.configure("TProgressbar", background='#FFC107')  # Yellow
            self.letter_label.config(fg='#FFC107')
        else:
            self.style.configure("TProgressbar", background='#F44336')  # Red
            self.letter_label.config(fg='#F44336')
        
        # Update history
        self.prediction_history.append((letter, confidence))
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
        
        # Update history text
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        history_str = ""
        for i, (l, c) in enumerate(reversed(self.prediction_history[-5:])):
            history_str += f"{l} ({c:.2f})  "
        self.history_text.insert(tk.END, history_str)
        self.history_text.config(state=tk.DISABLED)
        
        # Update scalogram visualization if provided
        if scalogram_img is not None:
            self.update_scalogram(scalogram_img)
            
    def update_scalogram(self, grid_img):
        """Update the scalogram visualization"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(grid_img, np.ndarray):
                # Scale to 0-255 range for display
                img_arr = (grid_img * 255).astype(np.uint8)
                img = Image.fromarray(img_arr)
                
                # Resize to fit canvas
                canvas_width = self.viz_canvas.winfo_width() or 300
                canvas_height = self.viz_canvas.winfo_height() or 150
                img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
                
                # Convert to PhotoImage for Tkinter
                photo = ImageTk.PhotoImage(img)
                
                # Update canvas
                self.viz_canvas.delete("all")
                self.viz_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.viz_canvas.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            print(f"Error updating scalogram: {e}")
    
    def update_status(self, status, color='white'):
        """Update the status message"""
        self.status_label.config(text=f"Status: {status}", fg=color)
    
    def check_queue(self):
        """Check for updates from the prediction thread"""
        # This will be called by the main thread to check for updates
        # from the prediction thread
        self.master.after(100, self.check_queue)

class SensorDataProcessor:
    def __init__(self, model_path, window_size=91, confidence_threshold=0.3, use_jit=True, gui=None, model_type='raw'):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.num_classes = 27
        self.class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','Rest','S','T','U','V','W','X','Y','Z']
        self.gui = gui
        self.model_type = model_type
        self.running = True
        
        # Initialize data buffers for each sensor
        self.sensor_buffers = [deque(maxlen=window_size) for _ in range(5)]
        self.data_queue = queue.Queue()
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_probs = None
        self.last_scalogram = None  # Store the last scalogram for visualization
        
        print(f"\nInitializing SensorDataProcessor with:")
        print(f"  Window size: {window_size}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Model type: {model_type}")
        
        if self.gui:
            self.gui.update_status("Initializing model...", "yellow")

        # Load model
        if self.model_type == 'scalogram':
            print(f"Loading scalogram model from {model_path}...")
            self.model = SimpleJetsonCNN(num_classes=self.num_classes)
            try:
                if not os.path.exists(model_path):
                    print(f"Error: Model file not found at {model_path}")
                    if self.gui:
                        self.gui.update_status(f"Error: Model file not found!", "red")
                    sys.exit(1)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Successfully loaded scalogram model from {model_path}")
                if self.gui:
                    self.gui.update_status(f"Loaded scalogram model", "green")
            except Exception as e:
                print(f"Error loading model state dict from {model_path}: {e}")
                if self.gui:
                    self.gui.update_status(f"Error loading model: {e}", "red")
                sys.exit(1)
        else:
            print(f"Loading raw signal model from {model_path}...")
            self.model = RawSignalCNN(num_classes=self.num_classes)
            try:
                if not os.path.exists(model_path):
                    print(f"Error: Model file not found at {model_path}")
                    if self.gui:
                        self.gui.update_status(f"Error: Model file not found!", "red")
                    sys.exit(1)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Successfully loaded raw signal model from {model_path}")
                if self.gui:
                    self.gui.update_status(f"Loaded raw signal model", "green")
            except Exception as e:
                print(f"Error loading model state dict from {model_path}: {e}")
                if self.gui:
                    self.gui.update_status(f"Error loading model: {e}", "red")
                sys.exit(1)
        self.model.eval()
        print("Model loaded and ready for inference")
        if self.gui:
            self.gui.update_status("Model loaded and ready", "green")

    def _load_regular_model(self, model_path):
        model = RawSignalCNN(num_classes=self.num_classes)
        try:
            # Check if the regular model file exists
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                if self.gui:
                    self.gui.update_status(f"Error: Model file not found!", "red")
                sys.exit(1)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Successfully loaded regular model from {model_path}")
            if self.gui:
                self.gui.update_status(f"Loaded regular model", "green")
        except Exception as e:
            print(f"Error loading model state dict from {model_path}: {e}")
            if self.gui:
                self.gui.update_status(f"Error loading model: {e}", "red")
            sys.exit(1)
        return model

    def start_c_process(self, i2c_device="/dev/i2c-1"):
        """Start the C process that reads sensor data"""
        # Construct path relative to this script's location
        c_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpu6050_reader")
        if not os.path.exists(c_executable):
            print(f"C executable not found at {c_executable}. Please compile NEW_C.c first.")
            print("Hint: gcc -o mpu6050_reader NEW_C.c -lm")
            if self.gui:
                self.gui.update_status(f"Error: C executable not found", "red")
            sys.exit(1)

        # Check if the process needs execution permissions
        if not os.access(c_executable, os.X_OK):
            print(f"Adding execute permissions to {c_executable}")
            try:
                os.chmod(c_executable, 0o755)
            except OSError as e:
                print(f"Error setting execute permission for {c_executable}: {e}")

        # Start the C process
        cmd = [c_executable, i2c_device]
        print(f"Starting C process: {' '.join(cmd)}")
        if self.gui:
            self.gui.update_status(f"Starting sensor data collection...", "yellow")
            
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line buffered
                universal_newlines=True,  # Text mode
                preexec_fn=os.setsid  # Create a new process group
            )
            print(f"Started sensor data collection process (PID: {self.process.pid})")
            if self.gui:
                self.gui.update_status(f"Sensor data collection started", "green")
        except Exception as e:
            print(f"Error starting C process: {e}")
            if self.gui:
                self.gui.update_status(f"Error starting sensor process: {e}", "red")
            sys.exit(1)

        # Start threads to read from stdout and stderr
        self.reader_thread = threading.Thread(target=self._read_process_output, name="StdoutReader")
        self.reader_thread.daemon = True
        self.reader_thread.start()
        print("Stdout reader thread started.")

        self.stderr_thread = threading.Thread(target=self._read_process_stderr, name="StderrReader")
        self.stderr_thread.daemon = True
        self.stderr_thread.start()
        print("Stderr reader thread started.")
        
    def _read_process_output(self):
        """Read data from the C process stdout"""
        print("Stdout reader thread started.")
        try:
            for line in self.process.stdout:
                if self.running:
                    try:
                                        # Parse the line from the C process
                        line = line.strip()
                        print(f"C output: {line}")  # Debug print
                        
                        # Try different formats of C output
                        parts = line.split()
                        # Direct format: "<sensor_idx> <gx> <gy> <gz> <ax> <ay> <az>"
                        if len(parts) >= 7 and parts[0].isdigit():
                            try:
                                sensor_idx = int(parts[0])
                                gx = float(parts[1])
                                gy = float(parts[2])
                                gz = float(parts[3])
                                ax = float(parts[4])
                                ay = float(parts[5])
                                az = float(parts[6])
                                print(f"Direct format: sensor {sensor_idx}: {gx}, {gy}, {gz}, {ax}, {ay}, {az}")
                            except ValueError:
                                print(f"Failed to parse direct format: {line}")
                                continue
                                
                                print(f"Parsed sensor {sensor_idx}: gx={gx}, gy={gy}, gz={gz}, ax={ax}, ay={ay}, az={az}")  # Debug print
                                
                                # Store data in the appropriate sensor buffer
                                if 0 <= sensor_idx < 5:  # We have 5 sensors (0-4)
                                    sensor_data = [gx, gy, gz, ax, ay, az]
                                    self.sensor_buffers[sensor_idx].append(sensor_data)
                                    
                                    # Print buffer sizes for debugging
                                    buffer_sizes = [len(buffer) for buffer in self.sensor_buffers]
                                    print(f"Buffer sizes: {buffer_sizes} (need {self.window_size} each)")
                                    
                                    # Check if all buffers are full enough for prediction
                                    self._check_and_process_data()
                    except Exception as e:
                        print(f"Error processing stdout line: {e}")
                else:
                    break
        except Exception as e:
            print(f"Error in stdout reader thread: {e}")
        print("Stdout reader thread finished.")
    
    def _read_process_stderr(self):
        """Read data from the C process stderr"""
        print("Stderr reader thread started.")
        try:
            for line in self.process.stderr:
                if not hasattr(self, 'running') or self.running:
                    print(f"[C Stderr]: {line.strip()}")
                else:
                    break
        except Exception as e:
            print(f"Error in stderr reader thread: {e}")
        print("Stderr reader thread finished.")

    def _check_and_process_data(self):
        """Check if all buffers are full and queue data for prediction."""
        # For debugging, always make a prediction even with partial data
        # Fill any empty buffers with zeros
        for i, buffer in enumerate(self.sensor_buffers):
            if len(buffer) < self.window_size:
                # Fill with zeros to reach window_size
                zeros_needed = self.window_size - len(buffer)
                print(f"Filling sensor {i} buffer with {zeros_needed} zero entries")
                for _ in range(zeros_needed):
                    buffer.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Create a numpy array from the buffers
        window_data = np.array([list(buffer) for buffer in self.sensor_buffers])
        print(f"Window data shape: {window_data.shape}")  # Debug print
        
        # Queue the data for processing
        try:
            self.data_queue.put(window_data, block=False)
            print(f"Data queued for prediction. Queue size: {self.data_queue.qsize()}")
            
            # Start the processing thread if not already running
            if not hasattr(self, 'processor_thread') or not self.processor_thread.is_alive():
                self.processor_thread = threading.Thread(target=self.process_data_loop, name="Processor")
                self.processor_thread.daemon = True
                self.processor_thread.start()
                print("Started prediction processing thread.")
        except queue.Full:
            # Queue is full, skip this window
            print("Queue full, skipping this window")
    
    def process_data_loop(self):
        """Main loop for processing data and making predictions"""
        print("Processing thread started.")
        print(f"Using model: {self.model.__class__.__name__}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Window size: {self.window_size}")
        
        while self.running:
            try:
                # Get data from queue with timeout
                try:
                    # Block for a short time to wait for data
                    window_data = self.data_queue.get(timeout=0.1)
                    print("\n*** Got data from queue for prediction ***")
                except queue.Empty:
                    if not self.running:
                        break
                    continue  # skip to next iteration

                # Process data based on model type
                if self.model_type == 'scalogram':
                    # window_data: (5, window, 6)
                    window_data = np.array(window_data)
                    grid_img = make_scalogram_grid(window_data)  # (H, W, 3)
                    img_pil = Image.fromarray((grid_img*255).astype(np.uint8))
                    img_pil = img_pil.resize((128, 128))
                    img_arr = np.array(img_pil).astype(np.float32) / 255.0
                    img_arr = (img_arr - 0.5) / 0.5  # Normalize to [-1, 1]
                    img_arr = np.transpose(img_arr, (2, 0, 1))  # (3, H, W)
                    img_tensor = torch.tensor(img_arr).unsqueeze(0)  # (1, 3, H, W)
                    with torch.no_grad():
                        logits = self.model(img_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_idx = int(np.argmax(probs))
                        confidence = float(probs[pred_idx])
                        self.last_probs = probs  # Store for visualization
                else:
                    # Raw signal model
                    window_data = np.array(window_data)
                    window_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)  # (1, 5, window, 6)
                    window_tensor = window_tensor.permute(0, 3, 1, 2).contiguous()  # (1, 6, 5, window)
                    window_tensor = window_tensor.unsqueeze(2)  # (1, 6, 1, 5, window)
                    with torch.no_grad():
                        logits = self.model(window_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_idx = int(np.argmax(probs))
                        confidence = float(probs[pred_idx])
                        self.last_probs = probs  # Store for visualization
                
                # Always update GUI with prediction regardless of confidence
                letter = self.class_names[pred_idx]
                print(f"Prediction: {letter} (Confidence: {confidence:.2f})")
                if self.gui:
                    # Store the scalogram for visualization
                    if self.model_type == 'scalogram':
                        self.last_scalogram = grid_img
                        self.gui.update_prediction(letter, confidence, grid_img)
                    else:
                        # For raw signal model, we can still visualize the raw data
                        self.last_scalogram = None
                        self.gui.update_prediction(letter, confidence)
                    
            except Exception as e:
                if self.running:
                    print(f"Error in processing loop: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1) # Avoid tight loop on error
        print("Processing thread finished.")

if __name__ == "__main__":
    print("\n===== Starting ASL Classifier GUI =====\n")
    parser = argparse.ArgumentParser(description="Real-time ASL classification GUI")
    parser.add_argument('--model', type=str, default="best_rawsignal_cnn.pth",
                        help="Path to model .pth file (raw or scalogram)")
    parser.add_argument('--model-type', type=str, default="raw", choices=["raw", "scalogram"],
                        help="Type of model to use: 'raw' for RawSignalCNN, 'scalogram' for SimpleJetsonCNN")
    parser.add_argument('--window', type=int, default=30,
                        help="Window size (samples)")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Confidence threshold for predictions")
    parser.add_argument('--i2c', type=str, default="/dev/i2c-1",
                        help="I2C device path for sensor reading")
    parser.add_argument('--debug', action='store_true',
                        help="Enable additional debug output")
    args = parser.parse_args()
    
    # Check for model file existence early
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Available model files:")
        for file in os.listdir("."):
            if file.endswith(".pth") or file.endswith(".pt"):
                print(f"  {file}")
        print("\nPlease specify a valid model file with --model")
        sys.exit(1)
    
    print(f"Using model: {args.model} (type: {args.model_type})")
    print(f"Window size: {args.window}, Threshold: {args.threshold}")
    print(f"I2C device: {args.i2c}")
    
    try:
        print("Creating Tkinter root window...")
        root = tk.Tk()
        print("Creating GUI...")
        gui = ASLClassifierGUI(root)
        gui.model_type = args.model_type
        gui.debug = args.debug
        
        print("Initializing sensor data processor...")
        processor = SensorDataProcessor(
            model_path=args.model,
            window_size=args.window,
            confidence_threshold=args.threshold,
            gui=gui,
            model_type=args.model_type
        )
        
        print("Starting C process for sensor reading...")
        processor.start_c_process(i2c_device=args.i2c)
        
        print("Starting GUI main loop...")
        root.mainloop()
        print("GUI main loop exited.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
