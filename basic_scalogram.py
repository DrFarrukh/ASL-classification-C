#!/usr/bin/env python3
"""
Basic Scalogram Viewer for ASL Classification
This script focuses on just showing scalograms with minimal dependencies.
"""
import sys
import os
import time
import subprocess
import threading
import argparse
from collections import deque
import tkinter as tk
from tkinter import Canvas

# Check if we have numpy and PIL
try:
    import numpy as np
    from PIL import Image, ImageTk
    HAS_NUMPY_PIL = True
except ImportError:
    print("WARNING: numpy or PIL not found. Using basic visualization.")
    HAS_NUMPY_PIL = False

class BasicScalogramViewer:
    def __init__(self, root, window_size=91):
        self.root = root
        self.window_size = window_size
        
        # Set up the UI
        root.title("ASL Basic Scalogram Viewer")
        root.geometry("800x600")
        root.configure(bg='black')
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        self.status = tk.Label(root, textvariable=self.status_var, 
                              fg='white', bg='black', font=('Arial', 12))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas for visualization
        self.canvas = Canvas(root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Sensor data buffers
        self.buffers = [deque(maxlen=window_size) for _ in range(5)]
        
        # For visualization
        self.colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        
    def update_status(self, text):
        """Update status text"""
        self.status_var.set(text)
        print(text)
        
    def add_sensor_data(self, sensor_idx, gx, gy, gz, ax, ay, az):
        """Add data for a specific sensor"""
        if 0 <= sensor_idx < 5:
            # Add new data point
            self.buffers[sensor_idx].append([gx, gy, gz, ax, ay, az])
            
            # Update status
            buffer_sizes = [len(b) for b in self.buffers]
            self.update_status(f"Buffer sizes: {buffer_sizes}")
            
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
            
        # Check if we have enough data
        if not all(len(buffer) > 0 for buffer in self.buffers):
            self.canvas.create_text(width/2, height/2, 
                                   text="Waiting for data from all sensors...", 
                                   fill='white', font=('Arial', 14))
            return
            
        if HAS_NUMPY_PIL:
            # Draw scalogram using numpy and PIL
            self._draw_scalogram(width, height)
        else:
            # Draw basic visualization
            self._draw_basic_visualization(width, height)
            
    def _draw_basic_visualization(self, width, height):
        """Draw a basic visualization of the sensor data"""
        # Draw sensor data as simple line plots
        for sensor_idx, buffer in enumerate(self.buffers):
            if not buffer:
                continue
                
            # Calculate vertical position for this sensor
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
                    scaled_y = base_y - (value / max_val) * 40
                    
                    points.append((scaled_x, scaled_y))
                
                # Draw the line if we have at least 2 points
                if len(points) >= 2:
                    self.canvas.create_line(points, fill=self.colors[i], width=2)
                    
                # Add label for this signal
                self.canvas.create_text(width - 100, base_y - 40 + i*15, 
                                       text=f"{name}: {data[-1]:.1f}", 
                                       fill=self.colors[i], anchor='e')
    
    def _draw_scalogram(self, width, height):
        """Draw a scalogram visualization using numpy and PIL"""
        try:
            # Convert buffers to numpy arrays
            data = []
            for buffer in self.buffers:
                if len(buffer) < self.window_size:
                    # Pad with zeros if needed
                    padding = [[0, 0, 0, 0, 0, 0] for _ in range(self.window_size - len(buffer))]
                    data.append(list(padding) + list(buffer))
                else:
                    data.append(list(buffer))
            
            # Create a numpy array from the data
            data_array = np.array(data)
            print(f"Data array shape: {data_array.shape}")
            
            # Create a simple visualization of the data
            # Each row is a sensor, each column is a time step
            # We'll create a grid where each cell's color represents the magnitude of the signal
            
            # Calculate the magnitude of the signal at each time step
            magnitudes = np.sqrt(np.sum(data_array**2, axis=2))
            print(f"Magnitudes shape: {magnitudes.shape}")
            
            # Normalize the magnitudes
            if np.max(magnitudes) > 0:
                magnitudes = magnitudes / np.max(magnitudes)
            
            # Create an RGB image
            img_array = np.zeros((5 * 50, self.window_size * 50, 3), dtype=np.uint8)
            
            # Fill the image with colors based on the magnitudes
            for i in range(5):  # 5 sensors
                for j in range(self.window_size):  # time steps
                    # Calculate the color based on the magnitude
                    r = int(255 * magnitudes[i, j])
                    g = int(255 * (1 - magnitudes[i, j]))
                    b = int(128)
                    
                    # Fill a 50x50 block with this color
                    img_array[i*50:(i+1)*50, j*50:(j+1)*50, 0] = r
                    img_array[i*50:(i+1)*50, j*50:(j+1)*50, 1] = g
                    img_array[i*50:(i+1)*50, j*50:(j+1)*50, 2] = b
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            
            # Resize to fit the canvas
            img = img.resize((width, height), Image.NEAREST)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Display on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep reference
            
            # Add labels
            for i in range(5):
                y_pos = i * height / 5 + height / 10
                self.canvas.create_text(10, y_pos, text=f"Sensor {i}", 
                                       fill='white', anchor='w')
        
        except Exception as e:
            print(f"Error drawing scalogram: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to basic visualization
            self._draw_basic_visualization(width, height)
    
    def generate_test_data(self):
        """Generate test data for visualization"""
        def _generate():
            t = 0
            while True:
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
    parser = argparse.ArgumentParser(description="Basic ASL Scalogram Viewer")
    parser.add_argument('--window', type=int, default=30, help="Window size (samples)")
    parser.add_argument('--i2c', type=str, default="/dev/i2c-1", help="I2C device path")
    parser.add_argument('--test', action='store_true', help="Use test data instead of real sensors")
    args = parser.parse_args()
    
    print("Starting Basic Scalogram Viewer")
    print(f"Window size: {args.window}")
    print(f"I2C device: {args.i2c}")
    print(f"Test mode: {args.test}")
    print(f"Using numpy/PIL: {HAS_NUMPY_PIL}")
    
    # Create the main window
    root = tk.Tk()
    app = BasicScalogramViewer(root, window_size=args.window)
    
    # Start data collection
    if args.test:
        app.update_status("Using test data")
        app.generate_test_data()
    else:
        if not app.start_c_process(i2c_device=args.i2c):
            app.update_status("Failed to start C process, using test data instead")
            app.generate_test_data()
    
    # Start the main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
