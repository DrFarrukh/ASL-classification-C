import numpy as np
import torch
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

class ScalogramViewer:
    def __init__(self, master):
        self.master = master
        master.title("ASL Scalogram Viewer")
        master.geometry("800x600")
        master.configure(bg='#2E2E2E')
        
        # Set up fonts
        self.title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=14)
        self.status_font = font.Font(family="Helvetica", size=12)
        
        # Create frames
        self.top_frame = tk.Frame(master, bg='#2E2E2E')
        self.top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Title
        self.title_label = tk.Label(
            self.top_frame, 
            text="ASL Scalogram Viewer", 
            font=self.title_font,
            fg='white',
            bg='#2E2E2E'
        )
        self.title_label.pack(pady=10)
        
        # Main scalogram visualization frame
        self.viz_frame = tk.Frame(master, bg='#2E2E2E')
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Add label for visualization
        self.viz_label = tk.Label(self.viz_frame, text="Scalogram Visualization:", font=self.label_font, fg='white', bg='#2E2E2E')
        self.viz_label.pack(anchor='w')
        
        # Add canvas for scalogram display
        self.viz_canvas = Canvas(self.viz_frame, bg='#1E1E1E', highlightthickness=0)
        self.viz_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Individual sensor visualizations
        self.sensor_frame = tk.Frame(master, bg='#2E2E2E')
        self.sensor_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Create a frame for each sensor
        self.sensor_canvases = []
        for i in range(5):
            frame = tk.Frame(self.sensor_frame, bg='#2E2E2E')
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            label = tk.Label(frame, text=f"Sensor {i}", font=self.label_font, fg='white', bg='#2E2E2E')
            label.pack(anchor='w')
            
            canvas = Canvas(frame, width=120, height=80, bg='#1E1E1E', highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            self.sensor_canvases.append(canvas)
        
        # Status frame
        self.status_frame = tk.Frame(master, bg='#2E2E2E')
        self.status_frame.pack(fill=tk.X, padx=20, pady=5)
        
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
        
        # Initialize data buffers
        self.window_size = 30
        self.sensor_buffers = [deque(maxlen=self.window_size) for _ in range(5)]
        self.running = True
        
        # Bind resize event to update canvas size
        self.viz_canvas.bind("<Configure>", self.on_resize)
        
    def on_resize(self, event):
        """Handle window resize events"""
        self.update_visualization()
        
    def update_status(self, status, color='white'):
        """Update the status message"""
        self.status_label.config(text=f"Status: {status}", fg=color)
        
    def update_visualization(self):
        """Update the scalogram visualization with current data"""
        try:
            # Check if we have data
            if not all(len(buffer) > 0 for buffer in self.sensor_buffers):
                return
                
            # Fill any empty buffers with zeros
            for i, buffer in enumerate(self.sensor_buffers):
                if len(buffer) < self.window_size:
                    zeros_needed = self.window_size - len(buffer)
                    for _ in range(zeros_needed):
                        buffer.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Create window data
            window_data = np.array([list(buffer) for buffer in self.sensor_buffers])
            
            # Generate scalogram grid
            grid_img = make_scalogram_grid(window_data)
            
            # Update main visualization
            self._update_canvas(self.viz_canvas, grid_img)
            
            # Update individual sensor visualizations
            for i, canvas in enumerate(self.sensor_canvases):
                if i < 5 and len(self.sensor_buffers[i]) > 0:
                    sensor_data = np.array(list(self.sensor_buffers[i]))
                    # Create a mini scalogram just for this sensor
                    gyro_scalo = rgb_scalogram_3axis(sensor_data[:,0], sensor_data[:,1], sensor_data[:,2])
                    self._update_canvas(canvas, gyro_scalo)
                    
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
            
    def _update_canvas(self, canvas, img_data):
        """Update a canvas with an image"""
        try:
            # Convert numpy array to PIL Image
            img_arr = (img_data * 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            
            # Resize to fit canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:  # Ensure valid dimensions
                img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
                
                # Convert to PhotoImage for Tkinter
                photo = ImageTk.PhotoImage(img)
                
                # Update canvas
                canvas.delete("all")
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            print(f"Error updating canvas: {e}")

class SensorDataProcessor:
    def __init__(self, window_size=30, gui=None):
        self.window_size = window_size
        self.gui = gui
        self.running = True
        
        # Initialize data buffers for each sensor
        self.sensor_buffers = gui.sensor_buffers if gui else [deque(maxlen=window_size) for _ in range(5)]
        
        print(f"\nInitializing SensorDataProcessor with:")
        print(f"  Window size: {window_size}")
        
        if self.gui:
            self.gui.update_status("Initializing sensor processor...", "yellow")
    
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
                        if "Raw C Output:" in line:
                            # Format: "Raw C Output: <sensor_idx> <gx> <gy> <gz> <ax> <ay> <az>"
                            parts = line.split()
                            if len(parts) >= 8:
                                sensor_idx = int(parts[3])
                                gx = float(parts[4])
                                gy = float(parts[5])
                                gz = float(parts[6])
                                ax = float(parts[7])
                                ay = float(parts[8]) if len(parts) > 8 else 0
                                az = float(parts[9]) if len(parts) > 9 else 0
                                
                                print(f"Parsed sensor {sensor_idx}: gx={gx}, gy={gy}, gz={gz}, ax={ax}, ay={ay}, az={az}")  # Debug print
                                
                                # Store data in the appropriate sensor buffer
                                if 0 <= sensor_idx < 5:  # We have 5 sensors (0-4)
                                    sensor_data = [gx, gy, gz, ax, ay, az]
                                    self.sensor_buffers[sensor_idx].append(sensor_data)
                                    
                                    # Update the visualization
                                    if self.gui:
                                        self.gui.master.after(10, self.gui.update_visualization)
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
                if self.running:
                    print(f"[C Stderr]: {line.strip()}")
                else:
                    break
        except Exception as e:
            print(f"Error in stderr reader thread: {e}")
        print("Stderr reader thread finished.")

def main():
    print("\n===== Starting ASL Scalogram Viewer =====\n")
    parser = argparse.ArgumentParser(description="ASL Scalogram Visualization")
    parser.add_argument('--window', type=int, default=30,
                        help="Window size (samples)")
    parser.add_argument('--i2c', type=str, default="/dev/i2c-1",
                        help="I2C device path for sensor reading")
    parser.add_argument('--debug', action='store_true',
                        help="Enable additional debug output")
    args = parser.parse_args()
    
    print(f"Window size: {args.window}")
    print(f"I2C device: {args.i2c}")
    
    try:
        print("Creating Tkinter root window...")
        root = tk.Tk()
        print("Creating GUI...")
        gui = ScalogramViewer(root)
        
        print("Initializing sensor data processor...")
        processor = SensorDataProcessor(
            window_size=args.window,
            gui=gui
        )
        
        print("Starting C process for sensor reading...")
        processor.start_c_process(i2c_device=args.i2c)
        
        # Set up periodic UI update
        def update_ui():
            gui.update_visualization()
            root.after(100, update_ui)
        
        root.after(100, update_ui)
        
        print("Starting GUI main loop...")
        root.mainloop()
        print("GUI main loop exited.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
