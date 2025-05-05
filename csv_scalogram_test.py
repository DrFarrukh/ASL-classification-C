#!/usr/bin/env python3
"""
CSV Scalogram Test
This script loads sensor data from a CSV file and visualizes it using the scalogram visualization.
"""
import numpy as np
import tkinter as tk
from tkinter import Canvas
import argparse
import os
import sys
from collections import deque
import csv
import time

# Check if we have PIL for image handling
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    print("WARNING: PIL not found. Using basic visualization.")
    HAS_PIL = False

# Import scalogram functions from basic_scalogram.py if available
try:
    from basic_scalogram import BasicScalogramViewer
    HAS_SCALOGRAM = True
except ImportError:
    print("WARNING: basic_scalogram.py not found. Using simplified visualization.")
    HAS_SCALOGRAM = False

class CSVScalogramTest:
    def __init__(self, csv_file, window_size=91):
        self.csv_file = csv_file
        self.window_size = window_size
        self.data = self.load_csv_data()
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title(f"CSV Scalogram Test - {os.path.basename(csv_file)}")
        self.root.geometry("800x600")
        
        if HAS_SCALOGRAM:
            # Use the BasicScalogramViewer
            self.viewer = BasicScalogramViewer(self.root, window_size=window_size)
            self.viewer.update_status(f"Loaded data from {csv_file}")
        else:
            # Create a simple viewer
            self.create_simple_viewer()
        
    def create_simple_viewer(self):
        """Create a simple viewer if BasicScalogramViewer is not available"""
        self.root.configure(bg='black')
        
        # Status label
        self.status_var = tk.StringVar(value=f"Loaded data from {self.csv_file}")
        self.status = tk.Label(self.root, textvariable=self.status_var, 
                              fg='white', bg='black', font=('Arial', 12))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas for visualization
        self.canvas = Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # For visualization
        self.colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    
    def load_csv_data(self):
        """Load data from CSV file"""
        print(f"Loading data from {self.csv_file}")
        
        # Initialize data structure
        data = {0: [], 1: [], 2: [], 3: [], 4: []}
        
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 8:  # Timestamp, sensor_idx, gx, gy, gz, ax, ay, az
                        timestamp = int(row[0])
                        sensor_idx = int(row[1])
                        gx = float(row[2])
                        gy = float(row[3])
                        gz = float(row[4])
                        ax = float(row[5])
                        ay = float(row[6])
                        az = float(row[7])
                        
                        # Store data
                        if 0 <= sensor_idx <= 4:
                            data[sensor_idx].append([gx, gy, gz, ax, ay, az])
            
            print(f"Loaded data for sensors: {list(data.keys())}")
            for sensor_idx, sensor_data in data.items():
                print(f"  Sensor {sensor_idx}: {len(sensor_data)} samples")
            
            return data
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return {0: [], 1: [], 2: [], 3: [], 4: []}
    
    def run_visualization(self):
        """Run the visualization"""
        if not any(len(data) > 0 for data in self.data.values()):
            print("No data to visualize!")
            return
        
        if HAS_SCALOGRAM:
            self.run_scalogram_visualization()
        else:
            self.run_simple_visualization()
    
    def run_scalogram_visualization(self):
        """Run visualization using BasicScalogramViewer"""
        # Process data in windows
        max_samples = max(len(data) for data in self.data.values())
        
        for i in range(0, max_samples, self.window_size // 2):  # 50% overlap
            # Clear previous data
            for sensor_idx in range(5):
                self.viewer.buffers[sensor_idx].clear()
            
            # Add data for this window
            for sensor_idx in range(5):
                if sensor_idx in self.data and self.data[sensor_idx]:
                    start_idx = min(i, len(self.data[sensor_idx]) - 1)
                    end_idx = min(i + self.window_size, len(self.data[sensor_idx]))
                    
                    for j in range(start_idx, end_idx):
                        self.viewer.add_sensor_data(
                            sensor_idx,
                            self.data[sensor_idx][j][0],  # gx
                            self.data[sensor_idx][j][1],  # gy
                            self.data[sensor_idx][j][2],  # gz
                            self.data[sensor_idx][j][3],  # ax
                            self.data[sensor_idx][j][4],  # ay
                            self.data[sensor_idx][j][5]   # az
                        )
            
            # Update visualization
            self.viewer.update_status(f"Window {i//self.window_size + 1}/{(max_samples + self.window_size - 1)//self.window_size}")
            self.viewer.update_visualization()
            
            # Process events
            self.root.update()
            time.sleep(0.5)  # Pause to see the visualization
    
    def run_simple_visualization(self):
        """Run a simple visualization if BasicScalogramViewer is not available"""
        # Process data in windows
        max_samples = max(len(data) for data in self.data.values())
        
        for i in range(0, max_samples, self.window_size // 2):  # 50% overlap
            # Clear canvas
            self.canvas.delete("all")
            
            # Get canvas dimensions
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            if width <= 1 or height <= 1:
                # Canvas not ready yet
                self.root.update()
                continue
            
            # Draw sensor data
            for sensor_idx in range(5):
                if sensor_idx in self.data and self.data[sensor_idx]:
                    # Calculate vertical position for this sensor
                    base_y = (sensor_idx + 1) * height / 6
                    
                    # Draw sensor label
                    self.canvas.create_text(10, base_y - 40, 
                                           text=f"Sensor {sensor_idx}", 
                                           fill='white', anchor='w')
                    
                    # Get data for this window
                    start_idx = min(i, len(self.data[sensor_idx]) - 1)
                    end_idx = min(i + self.window_size, len(self.data[sensor_idx]))
                    window_data = self.data[sensor_idx][start_idx:end_idx]
                    
                    # Draw each signal component
                    for j, name in enumerate(['gx', 'gy', 'gz', 'ax', 'ay', 'az']):
                        points = []
                        
                        # Scale the data to fit in the plot
                        data = [point[j] for point in window_data]
                        if not data:
                            continue
                            
                        # Normalize data for display
                        max_val = max(abs(max(data)), abs(min(data)))
                        if max_val == 0:
                            max_val = 1  # Avoid division by zero
                        
                        # Calculate points for the line
                        for x, value in enumerate(data):
                            # Scale x to fit width
                            scaled_x = x * width / len(data)
                            
                            # Scale y to fit height (centered around base_y)
                            scaled_y = base_y - (value / max_val) * 40
                            
                            points.append((scaled_x, scaled_y))
                        
                        # Draw the line if we have at least 2 points
                        if len(points) >= 2:
                            self.canvas.create_line(points, fill=self.colors[j % len(self.colors)], width=2)
            
            # Update status
            self.status_var.set(f"Window {i//self.window_size + 1}/{(max_samples + self.window_size - 1)//self.window_size}")
            
            # Process events
            self.root.update()
            time.sleep(0.5)  # Pause to see the visualization
    
    def run(self):
        """Run the application"""
        try:
            # Run visualization
            self.run_visualization()
            
            # Start main loop
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Exiting...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="CSV Scalogram Test")
    parser.add_argument('csv_file', type=str, help="Path to CSV file")
    parser.add_argument('--window', type=int, default=91, help="Window size (samples)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)
    
    # Create and run the application
    app = CSVScalogramTest(args.csv_file, window_size=args.window)
    app.run()

if __name__ == "__main__":
    main()
