import numpy as np
import tkinter as tk
from tkinter import ttk, Canvas
import pywt
from PIL import Image, ImageTk
from matplotlib.colors import Normalize
import argparse
import time
import threading
import os
import sys
import subprocess
from collections import deque

# Simple class to store and visualize sensor data
class SensorDataVisualizer:
    def __init__(self, root, window_size=30):
        self.root = root
        self.window_size = window_size
        
        # Create a simple UI
        root.title("ASL Sensor Data Viewer")
        root.geometry("800x600")
        root.configure(bg='black')
        
        # Create canvas for visualization
        self.canvas = Canvas(root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        self.status = tk.Label(root, textvariable=self.status_var, 
                              fg='white', bg='black', font=('Arial', 12))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Sensor data buffers - one for each sensor (0-4)
        self.buffers = [[] for _ in range(5)]
        
        # For visualization
        self.plot_height = 100
        self.colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        
        # Start with test data if requested
        self.test_mode = False
        
    def update_status(self, text):
        """Update status text"""
        self.status_var.set(text)
        
    def add_sensor_data(self, sensor_idx, gx, gy, gz, ax, ay, az):
        """Add data for a specific sensor"""
        if 0 <= sensor_idx < 5:
            # Store only the last window_size points
            if len(self.buffers[sensor_idx]) >= self.window_size:
                self.buffers[sensor_idx].pop(0)
            
            # Add new data point
            self.buffers[sensor_idx].append([gx, gy, gz, ax, ay, az])
            
            # Update visualization
            self.update_visualization()
            
    def update_visualization(self):
        """Update the visualization with current data"""
        # Clear canvas
        self.canvas.delete("all")
        
        # Get canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            # Canvas not ready yet
            return
            
        # Draw sensor data
        for sensor_idx, buffer in enumerate(self.buffers):
            if not buffer:
                continue
                
            # Calculate vertical position for this sensor's data
            base_y = (sensor_idx + 1) * height / 6
            
            # Draw sensor label
            self.canvas.create_text(10, base_y - 40, 
                                   text=f"Sensor {sensor_idx}", 
                                   fill='white', anchor='w')
            
            # Draw each signal component
            for i, name in enumerate(['gx', 'gy', 'gz', 'ax', 'ay', 'az']):
                points = []
                
                # Scale the data to fit in the plot
                data = [point[i] for point in buffer]
                if not data:
                    continue
                    
                # Normalize data for display
                max_val = max(abs(max(data)), abs(min(data)))
                if max_val == 0:
                    max_val = 1  # Avoid division by zero
                
                # Calculate points for the line
                for x, value in enumerate(data):
                    # Scale x to fit width
                    scaled_x = x * width / self.window_size
                    
                    # Scale y to fit height (centered around base_y)
                    scaled_y = base_y - (value / max_val) * (self.plot_height / 2)
                    
                    points.append((scaled_x, scaled_y))
                
                # Draw the line if we have at least 2 points
                if len(points) >= 2:
                    self.canvas.create_line(points, fill=self.colors[i], width=2)
                    
                # Add label for this signal
                self.canvas.create_text(width - 100, base_y - 40 + i*15, 
                                       text=f"{name}: {data[-1]:.1f}", 
                                       fill=self.colors[i], anchor='e')
        
        # Update status with buffer sizes
        buffer_sizes = [len(b) for b in self.buffers]
        self.update_status(f"Buffer sizes: {buffer_sizes}")
        
    def generate_test_data(self):
        """Generate test data for visualization"""
        self.test_mode = True
        
        def _generate():
            t = 0
            while self.test_mode:
                for sensor_idx in range(5):
                    # Generate sine waves with different frequencies
                    gx = 500 * np.sin(t + sensor_idx * 0.5)
                    gy = 500 * np.cos(t + sensor_idx * 0.5)
                    gz = 500 * np.sin(t + sensor_idx * 0.5 + np.pi/4)
                    ax = 15000 * np.cos(t + sensor_idx * 0.5 + np.pi/3)
                    ay = 15000 * np.sin(t + sensor_idx * 0.5 + np.pi/2)
                    az = 15000 * np.cos(t + sensor_idx * 0.5 + np.pi/6)
                    
                    # Add to buffer
                    self.add_sensor_data(sensor_idx, gx, gy, gz, ax, ay, az)
                
                # Increment time
                t += 0.1
                time.sleep(0.1)
        
        # Start test data generation in a thread
        threading.Thread(target=_generate, daemon=True).start()
        
    def start_c_process(self, i2c_device="/dev/i2c-1"):
        """Start the C process to read sensor data"""
        # Find the C executable
        c_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpu6050_reader")
        if not os.path.exists(c_executable):
            self.update_status(f"Error: C executable not found at {c_executable}")
            return False
            
        # Check permissions
        if not os.access(c_executable, os.X_OK):
            try:
                os.chmod(c_executable, 0o755)
                self.update_status(f"Added execute permission to {c_executable}")
            except Exception as e:
                self.update_status(f"Error setting permissions: {e}")
                return False
                
        # Start the C process
        cmd = [c_executable, i2c_device]
        self.update_status(f"Starting C process: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid
            )
            
            # Start reader threads
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            
            self.update_status(f"C process started (PID: {self.process.pid})")
            return True
        except Exception as e:
            self.update_status(f"Error starting C process: {e}")
            return False
            
    def _read_stdout(self):
        """Read data from the C process stdout"""
        for line in self.process.stdout:
            try:
                line = line.strip()
                print(f"C output: {line}")
                
                # Parse the line: "<sensor_idx> <gx> <gy> <gz> <ax> <ay> <az>"
                parts = line.split()
                if len(parts) >= 7 and parts[0].isdigit():
                    sensor_idx = int(parts[0])
                    gx = float(parts[1])
                    gy = float(parts[2])
                    gz = float(parts[3])
                    ax = float(parts[4])
                    ay = float(parts[5])
                    az = float(parts[6])
                    
                    # Add to buffer
                    self.add_sensor_data(sensor_idx, gx, gy, gz, ax, ay, az)
            except Exception as e:
                print(f"Error parsing line: {e}")
                
    def _read_stderr(self):
        """Read data from the C process stderr"""
        for line in self.process.stderr:
            print(f"[C stderr] {line.strip()}")

def main():
    parser = argparse.ArgumentParser(description="Simple ASL Sensor Data Viewer")
    parser.add_argument('--window', type=int, default=30, help="Window size (samples)")
    parser.add_argument('--i2c', type=str, default="/dev/i2c-1", help="I2C device path")
    parser.add_argument('--test', action='store_true', help="Use test data instead of real sensors")
    args = parser.parse_args()
    
    # Create the main window
    root = tk.Tk()
    app = SensorDataVisualizer(root, window_size=args.window)
    
    # Start data collection
    if args.test:
        app.update_status("Using test data")
        app.generate_test_data()
    else:
        if not app.start_c_process(i2c_device=args.i2c):
            app.update_status("Failed to start C process, using test data instead")
            app.generate_test_data()
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
