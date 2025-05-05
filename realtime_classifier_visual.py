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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pywt

# --- Raw Signal Model definition ---
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

class SensorDataProcessor:
    def __init__(self, model_path, window_size=91, confidence_threshold=0.6, use_jit=True, visualize=True):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.num_classes = 27
        self.class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','Rest','S','T','U','V','W','X','Y','Z']
        self.visualize = visualize

        # Initialize data buffers for each sensor
        self.sensor_buffers = [deque(maxlen=window_size) for _ in range(5)]
        self.data_queue = queue.Queue()
        self.running = True
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_report_time = 0 # Initialize last report time

        # For visualization
        if self.visualize:
            self.timeline_preds = []
            self.timeline_times = []
            self.timeline_probs = []
            plt.ion()
            self.fig, (self.ax_img, self.ax_bar, self.ax_timeline) = plt.subplots(1, 3, figsize=(18, 5))
            plt.tight_layout()

        # Load model
        print(f"Loading model from {model_path}...")
        # Construct potential JIT model path
        model_jit_path = model_path.replace('.pth', '_jit.pt')
        if use_jit and os.path.exists(model_jit_path):
            try:
                self.model = torch.jit.load(model_jit_path)
                print(f"Successfully loaded JIT model from {model_jit_path}")
            except Exception as e:
                print(f"Failed to load JIT model from {model_jit_path}: {e}")
                print("Falling back to regular model loading...")
                self.model = self._load_regular_model(model_path)
        else:
            if use_jit:
                 print(f"JIT model not found at {model_jit_path}. Loading regular model.")
            else:
                 print("JIT model not requested. Loading regular model.")
            self.model = self._load_regular_model(model_path)

        self.model.eval()
        print("Model loaded and ready for inference")

    def _load_regular_model(self, model_path):
        model = RawSignalCNN(num_classes=self.num_classes)
        try:
            # Check if the regular model file exists
            if not os.path.exists(model_path):
                 print(f"Error: Model file not found at {model_path}")
                 sys.exit(1)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Successfully loaded regular model from {model_path}")
        except Exception as e:
             print(f"Error loading model state dict from {model_path}: {e}")
             sys.exit(1)
        return model

    def start_c_process(self, i2c_device="/dev/i2c-1"):
        """Start the C process that reads sensor data"""
        # Construct path relative to this script's location
        c_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpu6050_reader")
        if not os.path.exists(c_executable):
            print(f"C executable not found at {c_executable}. Please compile NEW_C.c first.")
            print("Hint: gcc -o mpu6050_reader NEW_C.c -lm")
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
        except Exception as e:
            print(f"Error starting C process: {e}")
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
        print("Waiting for C process to initialize...")
        try:
            for line in self.process.stdout:
                if not self.running:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse the line
                    # Expected format: "sensor_idx gx gy gz ax ay az"
                    parts = line.split()
                    if len(parts) != 7:
                        print(f"Invalid data format: {line}")
                        continue

                    sensor_idx = int(parts[0])
                    if sensor_idx < 0 or sensor_idx >= 5:
                        print(f"Invalid sensor index: {sensor_idx}")
                        continue

                    # Parse the values
                    values = [float(x) for x in parts[1:]]
                    # Order: gx gy gz ax ay az
                    
                    # Add the values to the appropriate buffer
                    self.sensor_buffers[sensor_idx].append(values)
                    
                    # Check if we have enough data for a prediction
                    self._check_and_process_data()
                    
                except Exception as e:
                    print(f"Error parsing data: {e}")
                    continue
        except Exception as e:
            if self.running:
                print(f"Error reading from C process: {e}")
        finally:
            print("Stdout reader thread finished.")

    def _read_process_stderr(self):
        """Read data from the C process stderr"""
        try:
            for line in self.process.stderr:
                if not self.running:
                    break
                
                line = line.strip()
                if line:
                    print(f"[C Stderr]: {line}")
                    
                    # Check for initialization message
                    if "initialized" in line.lower() or "connected" in line.lower():
                        print("C process initialized successfully.")
        except Exception as e:
            if self.running:
                print(f"Error reading stderr: {e}")
        finally:
            # Check if there's any final stderr output
            if hasattr(self, 'process') and self.process.poll() is not None:
                remaining = self.process.stderr.read()
                if remaining:
                    print(f"[C Stderr - Final]: {remaining.strip()}")
            print("Stderr reader thread finished.")

    def _check_and_process_data(self):
        """Check if all buffers are full and queue data for prediction."""
        # Check if all buffers have enough data
        if all(len(buffer) >= self.window_size for buffer in self.sensor_buffers):
            # Create a numpy array from the buffers
            window_data = np.array([list(buffer) for buffer in self.sensor_buffers])
            
            # Queue the data for processing
            try:
                self.data_queue.put(window_data, block=False)
            except queue.Full:
                # Queue is full, skip this window
                pass
            
            # --- Overlapping vs Non-overlapping windows --- 
            # The deque with maxlen handles overlapping windows automatically.
            # If non-overlapping windows are needed, clear the buffers here:
            # for buffer in self.sensor_buffers:
            #    buffer.clear()
            # -------------------------------------------------

    def process_data_loop(self):
        """Main loop for processing data and making predictions"""
        print("Processing thread started.")
        while self.running:
            try:
                # Get data from queue with timeout
                try:
                    # Block for a short time to wait for data
                    window_data = self.data_queue.get(timeout=0.1)
                except queue.Empty:
                    # No data available, continue loop
                    # Check if we should still be running
                    if not self.running:
                         break
                    continue

                # Prepare data for model
                # Expected input shape: (1, 6, 5, 1, window_size)
                x = window_data.astype(np.float32)  # (5, window, 6)
                x = np.transpose(x, (2, 0, 1))  # (6, 5, window)
                x = np.expand_dims(x, axis=2)  # (6, 5, 1, window)
                x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, 6, 5, 1, window)

                # Make prediction
                with torch.no_grad():
                    logits = self.model(x_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = probs[pred_idx]

                # Only report if confidence is above threshold and prediction changed or enough time passed
                current_time = time.time()
                # Report if:
                # - Confidence is high enough AND
                # - (Prediction is different OR it's been > 1.5s since last report of *any* prediction)
                if confidence >= self.confidence_threshold and \
                   (pred_idx != self.last_prediction or current_time - self.last_report_time > 1.5):
                        self.last_prediction = pred_idx
                        self.last_confidence = confidence
                        self.last_report_time = current_time
                        print(f"Prediction: {self.class_names[pred_idx]} (Confidence: {confidence:.2f})")
                
                # Update visualization regardless of confidence
                if self.visualize:
                    self._update_visualization(window_data, probs, pred_idx, current_time)

            except Exception as e:
                if self.running:
                    print(f"Error in processing loop: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1) # Avoid tight loop on error
        print("Processing thread finished.")

    def _update_visualization(self, window_data, probs, pred_idx, current_time):
        """Update the visualization with the latest data and prediction"""
        try:
            # Create scalogram grid
            grid_img = make_scalogram_grid(window_data)
            
            # Update image plot
            self.ax_img.clear()
            self.ax_img.imshow(grid_img)
            self.ax_img.set_title('Sensor Data Scalogram')
            self.ax_img.axis('off')
            
            # Update bar chart
            self.ax_bar.clear()
            bar_container = self.ax_bar.bar(range(len(self.class_names)), probs, color='skyblue')
            self.ax_bar.set_ylim(0, 1)
            self.ax_bar.set_title('Class Probabilities')
            self.ax_bar.set_ylabel('Probability')
            self.ax_bar.set_xticks(range(len(self.class_names)))
            self.ax_bar.set_xticklabels(self.class_names, rotation=90)
            self.ax_bar.axhline(self.confidence_threshold, color='red', linestyle='--', linewidth=0.8)
            self.ax_bar.bar(pred_idx, probs[pred_idx], color='orange')
            
            # Add text label for the prediction
            if probs[pred_idx] >= self.confidence_threshold:
                self.ax_bar.text(pred_idx, probs[pred_idx]+0.03, 
                                f'{self.class_names[pred_idx]}\n{probs[pred_idx]:.2f}', 
                                ha='center', color='black')
            
            # Update timeline
            self.timeline_preds.append(pred_idx)
            self.timeline_times.append(current_time)
            self.timeline_probs.append(probs[pred_idx])
            
            # Keep only the last 100 predictions for the timeline
            if len(self.timeline_preds) > 100:
                self.timeline_preds = self.timeline_preds[-100:]
                self.timeline_times = self.timeline_times[-100:]
                self.timeline_probs = self.timeline_probs[-100:]
            
            # Plot timeline
            self.ax_timeline.clear()
            
            # Create a color array based on confidence
            colors = ['red' if p < self.confidence_threshold else 'green' for p in self.timeline_probs]
            
            # Plot each point with its color
            for i in range(len(self.timeline_times)):
                self.ax_timeline.scatter(i, self.timeline_preds[i], color=colors[i], s=20)
            
            # Connect the points with a line
            self.ax_timeline.plot(range(len(self.timeline_times)), self.timeline_preds, 
                                 color='blue', alpha=0.5, linestyle='-')
            
            self.ax_timeline.set_yticks(range(len(self.class_names)))
            self.ax_timeline.set_yticklabels(self.class_names)
            self.ax_timeline.set_xlabel('Time')
            self.ax_timeline.set_ylabel('Predicted Class')
            self.ax_timeline.set_title('Prediction Timeline')
            self.ax_timeline.grid(True, axis='y', linestyle='--', alpha=0.5)
            self.ax_timeline.set_ylim(-1, len(self.class_names))
            
            # Set x-axis limits to show only the last 100 predictions
            self.ax_timeline.set_xlim(0, len(self.timeline_times))
            
            # Update the figure
            plt.tight_layout()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"Error updating visualization: {e}")

    def start(self, i2c_device="/dev/i2c-1"):
        """Start the data collection and processing"""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start the C process
        self.start_c_process(i2c_device)

        # Start the processing thread
        self.processor_thread = threading.Thread(target=self.process_data_loop, name="Processor")
        self.processor_thread.daemon = True
        self.processor_thread.start()

        print("Real-time classification starting. Press Ctrl+C to stop.")

        try:
            # Monitor the C process
            while self.running:
                # Check if the C process is still running
                if self.process.poll() is not None:
                    exit_code = self.process.returncode
                    print(f"C process exited unexpectedly. Stopping...")
                    print(f"C process exit code: {exit_code}")
                    self.running = False
                    self.stop()
                    break
                
                # Sleep to avoid tight loop
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")
            self.running = False
            self.stop()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.running = False
            self.stop()

    def _signal_handler(self, sig, frame):
        print(f"\nReceived signal {sig}. Stopping...")
        self.running = False
        self.stop()

    def stop(self):
        """Stop all threads and processes gracefully"""
        print("Stopping real-time classification...")
        self.running = False

        # Close the visualization window if it's open
        if self.visualize:
            plt.close(self.fig)
            
        # Terminate the C process if it's still running
        if hasattr(self, 'process') and self.process.poll() is None:
            print("Terminating C process...")
            # Send SIGTERM to the process group
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception as e:
                print(f"Error sending SIGTERM to process group: {e}")
                
            # Fallback: terminate the process directly
            self.process.terminate() 
            try:
                # Wait for a short period for graceful termination
                self.process.wait(timeout=2.0)
                print("C process terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("C process did not terminate gracefully after SIGTERM, sending SIGKILL...")
                self.process.kill() # Force kill if needed
                try:
                     self.process.wait(timeout=1.0) # Wait briefly for kill
                     print("C process killed.")
                except subprocess.TimeoutExpired:
                     print("Warning: C process did not respond to SIGKILL.")
            except Exception as e:
                 print(f"Error during C process termination: {e}")
        elif hasattr(self, 'process'):
             # print("C process already terminated.")
             pass # Process already finished

        # Wait for threads to finish
        print("Waiting for threads to finish...")
        threads_to_join = []
        if hasattr(self, 'reader_thread') and self.reader_thread.is_alive():
             threads_to_join.append(self.reader_thread)
        if hasattr(self, 'stderr_thread') and self.stderr_thread.is_alive():
             threads_to_join.append(self.stderr_thread)
        if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
             threads_to_join.append(self.processor_thread)
             
        for thread in threads_to_join:
             try:
                  thread.join(timeout=1.5)
                  # print(f"Thread {thread.name} finished.")
             except Exception as e:
                  print(f"Error joining thread {thread.name}: {e}")
                  
        # Check if threads are still alive after join timeout
        # alive_threads = [t.name for t in threads_to_join if t.is_alive()]
        # if alive_threads:
        #      print(f"Warning: The following threads did not finish gracefully: {', '.join(alive_threads)}")

        print("Real-time classification stopped.")
        # Use os._exit(0) for a more immediate exit in case of lingering threads
        # sys.exit(0) # Use standard exit
        os._exit(0) # Force exit if necessary

def main():
    parser = argparse.ArgumentParser(description="Real-time ASL classification from MPU6050 sensors")
    parser.add_argument('--model', type=str, default="best_rawsignal_cnn.pth",
                        help="Path to model .pth file (JIT version '<model_name>_jit.pt' will be checked if --use-jit is set)")
    parser.add_argument('--window', type=int, default=91,
                        help="Window size (samples)")
    parser.add_argument('--threshold', type=float, default=0.6,
                        help="Confidence threshold for predictions")
    parser.add_argument('--i2c', type=str, default="/dev/i2c-1",
                        help="I2C device path")
    parser.add_argument('--use-jit', action='store_true',
                        help="Use TorchScript JIT model if available (looks for <model_name>_jit.pt)")
    parser.add_argument('--no-visual', action='store_true',
                        help="Disable visualization")
    args = parser.parse_args()

    # Ensure the model path exists before starting
    model_exists = os.path.exists(args.model)
    jit_model_path = args.model.replace('.pth', '_jit.pt')
    jit_model_exists = os.path.exists(jit_model_path)

    if not model_exists and not (args.use_jit and jit_model_exists):
         print(f"Error: Model file not found!")
         print(f"Checked for: {args.model}")
         if args.use_jit:
              print(f"Checked for JIT model: {jit_model_path}")
         sys.exit(1)
         
    if args.use_jit and not jit_model_exists:
        print(f"Warning: --use-jit specified, but JIT model '{jit_model_path}' not found. Will use regular model '{args.model}'.")
        
    # Check if PyWavelets is installed
    try:
        import pywt
    except ImportError:
        print("PyWavelets is required for visualization. Please install it with:")
        print("pip install PyWavelets")
        if not args.no_visual:
            print("Running without visualization...")
            args.no_visual = True

    processor = SensorDataProcessor(
        model_path=args.model,
        window_size=args.window,
        confidence_threshold=args.threshold,
        use_jit=args.use_jit,
        visualize=not args.no_visual
    )

    processor.start(i2c_device=args.i2c)

if __name__ == "__main__":
    main()
