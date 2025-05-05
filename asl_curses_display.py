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
import curses
from collections import deque

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

class SensorDataProcessor:
    def __init__(self, model_path, window_size=91, confidence_threshold=0.3, use_jit=True, stdscr=None):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.num_classes = 27
        self.class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','Rest','S','T','U','V','W','X','Y','Z']
        self.stdscr = stdscr
        
        # Initialize curses colors if we have a screen
        if self.stdscr:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)  # Green for high confidence
            curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Yellow for medium confidence
            curses.init_pair(3, curses.COLOR_RED, -1)    # Red for low confidence
            curses.init_pair(4, curses.COLOR_CYAN, -1)   # Cyan for headers
            curses.init_pair(5, curses.COLOR_WHITE, -1)  # White for normal text
            self.stdscr.clear()
            
        self.log(f"Initializing SensorDataProcessor with:")
        self.log(f"  Window size: {window_size}")
        self.log(f"  Confidence threshold: {confidence_threshold}")
        self.log(f"  Use JIT: {use_jit}")

        # Initialize data buffers for each sensor
        self.sensor_buffers = [deque(maxlen=window_size) for _ in range(5)]
        self.data_queue = queue.Queue()
        self.running = True
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_report_time = 0 # Initialize last report time
        
        # For tracking predictions
        self.recent_predictions = []
        self.recent_confidences = []

        # Load model
        self.log(f"Loading model from {model_path}...")
        # Construct potential JIT model path
        model_jit_path = model_path.replace('.pth', '_jit.pt')
        if use_jit and os.path.exists(model_jit_path):
            try:
                self.model = torch.jit.load(model_jit_path)
                self.log(f"Successfully loaded JIT model from {model_jit_path}")
            except Exception as e:
                self.log(f"Failed to load JIT model from {model_jit_path}: {e}")
                self.log("Falling back to regular model loading...")
                self.model = self._load_regular_model(model_path)
        else:
            if use_jit:
                 self.log(f"JIT model not found at {model_jit_path}. Loading regular model.")
            else:
                 self.log("JIT model not requested. Loading regular model.")
            self.model = self._load_regular_model(model_path)

        self.model.eval()
        self.log("Model loaded and ready for inference")

    def log(self, message):
        """Log a message to the screen or stdout"""
        if self.stdscr:
            # Add to a log buffer that we could display if needed
            pass
        else:
            print(message)

    def _load_regular_model(self, model_path):
        model = RawSignalCNN(num_classes=self.num_classes)
        try:
            # Check if the regular model file exists
            if not os.path.exists(model_path):
                 self.log(f"Error: Model file not found at {model_path}")
                 sys.exit(1)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.log(f"Successfully loaded regular model from {model_path}")
        except Exception as e:
             self.log(f"Error loading model state dict from {model_path}: {e}")
             sys.exit(1)
        return model

    def start_c_process(self, i2c_device="/dev/i2c-1"):
        """Start the C process that reads sensor data"""
        # Construct path relative to this script's location
        c_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpu6050_reader")
        if not os.path.exists(c_executable):
            self.log(f"C executable not found at {c_executable}. Please compile NEW_C.c first.")
            self.log("Hint: gcc -o mpu6050_reader NEW_C.c -lm")
            sys.exit(1)

        # Check if the process needs execution permissions
        if not os.access(c_executable, os.X_OK):
            self.log(f"Adding execute permissions to {c_executable}")
            try:
                os.chmod(c_executable, 0o755)
            except OSError as e:
                self.log(f"Error setting execute permission for {c_executable}: {e}")

        # Start the C process
        cmd = [c_executable, i2c_device]
        self.log(f"Starting C process: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line buffered
                universal_newlines=True,  # Text mode
                preexec_fn=os.setsid  # Create a new process group
            )
            self.log(f"Started sensor data collection process (PID: {self.process.pid})")
        except Exception as e:
            self.log(f"Error starting C process: {e}")
            sys.exit(1)

        # Start threads to read from stdout and stderr
        self.reader_thread = threading.Thread(target=self._read_process_output, name="StdoutReader")
        self.reader_thread.daemon = True
        self.reader_thread.start()
        self.log("Stdout reader thread started.")

        self.stderr_thread = threading.Thread(target=self._read_process_stderr, name="StderrReader")
        self.stderr_thread.daemon = True
        self.stderr_thread.start()
        self.log("Stderr reader thread started.")

    def _read_process_output(self):
        """Read data from the C process stdout"""
        self.log("Stdout reader thread started.")
        # Ensure process and stdout exist before entering loop
        if not hasattr(self, 'process') or not self.process.stdout:
             self.log("Error: C process or stdout not available for reading.")
             self.running = False # Signal main loop to stop
             return

        while self.running:
            try:
                # Check if process has exited before reading
                if self.process.poll() is not None:
                    break

                line = self.process.stdout.readline()
                if not line:
                    # Check again if process exited after readline returned empty
                    if self.process.poll() is not None:
                        break
                    # If process is still running, readline might timeout or encounter EOF temporarily
                    time.sleep(0.01) # Small sleep to avoid busy-waiting
                    continue

                line = line.strip()
                if not line:
                    continue

                # Parse the line: sensor_index ax ay az gx gy gz
                parts = line.split()
                if len(parts) != 7:
                    self.log(f"Skipping invalid line (expected 7 parts, got {len(parts)}): {line}")
                    continue

                # Use try-except for robust parsing
                try:
                    sensor_idx = int(parts[0])
                    # Validate sensor index range
                    if not (0 <= sensor_idx < 5):
                        self.log(f"Skipping invalid sensor index ({sensor_idx}): {line}")
                        continue

                    # Extract sensor values in the order they appear in the C output
                    # C Output: sensor_idx gx gy gz ax ay az
                    gx, gy, gz, ax, ay, az = map(float, parts[1:])

                    # Create the data point - already in the correct order for the model
                    # Model Input Order: gx, gy, gz, ax, ay, az
                    data_point = [gx, gy, gz, ax, ay, az]

                except ValueError as e:
                    self.log(f"Error parsing numeric values in line '{line}': {e}")
                    continue # Skip this line

                # Add to the appropriate sensor buffer
                self.sensor_buffers[sensor_idx].append(data_point)

                # Check if we have enough data for inference (moved outside parsing try-except)
                self._check_and_process_data()

                # Slow down: 90 samples in 5 seconds ≈ 0.055 sec/sample
                time.sleep(0.055)

            except Exception as e:
                # Catch unexpected errors during the read/process loop
                if self.running: # Avoid printing errors during shutdown
                     self.log(f"Unexpected error in stdout reader thread: {e}")
                     import traceback
                     traceback.print_exc()
                time.sleep(0.1) # Avoid tight loop on continuous error

        self.log("Stdout reader thread finished.")

    def _read_process_stderr(self):
        """Read data from the C process stderr"""
        try:
            for line in self.process.stderr:
                if not self.running:
                    break
                
                line = line.strip()
                if line:
                    # Just log C stderr messages at debug level
                    pass
                    
                    # Check for initialization message
                    if "initialized" in line.lower() or "connected" in line.lower():
                        self.log("C process initialized successfully.")
        except Exception as e:
            if self.running:
                self.log(f"Error reading stderr: {e}")
        finally:
            # Check if there's any final stderr output
            if hasattr(self, 'process') and self.process.poll() is not None:
                remaining = self.process.stderr.read()
                if remaining:
                    self.log(f"[C Stderr - Final]: {remaining.strip()}")
            self.log("Stderr reader thread finished.")

    def _check_and_process_data(self):
        """Check if all buffers are full and queue data for prediction."""
        # Get current buffer sizes
        buffer_sizes = [len(buffer) for buffer in self.sensor_buffers]
        
        # Check if all buffers have enough data
        if all(size >= self.window_size for size in buffer_sizes):
            # Create a numpy array from the buffers
            window_data = np.array([list(buffer) for buffer in self.sensor_buffers])
            
            # Queue the data for processing
            try:
                self.data_queue.put(window_data, block=False)
            except queue.Full:
                # Queue is full, skip this window
                pass

    def update_display(self, pred_idx, confidence):
        """Update the curses display with the prediction"""
        if not self.stdscr:
            return
            
        try:
            self.stdscr.clear()
            height, width = self.stdscr.getmaxyx()
            
            # Title and time on the same line
            title = "ASL CLASSIFIER"
            time_str = time.strftime("%H:%M:%S")
            self.stdscr.addstr(0, 2, title, curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(0, width - len(time_str) - 2, time_str, curses.color_pair(5))
            
            # Divider
            self.stdscr.addstr(1, 0, "=" * (width-1), curses.color_pair(5))
            
            # Current prediction (large letter) and confidence on the same line
            letter = self.class_names[pred_idx]
            
            # Choose color based on confidence
            if confidence >= 0.7:
                color = curses.color_pair(1)  # Green
            elif confidence >= 0.4:
                color = curses.color_pair(2)  # Yellow
            else:
                color = curses.color_pair(3)  # Red
                
            # Display letter and confidence on the same line
            self.stdscr.addstr(3, 2, "PREDICTION:", curses.color_pair(5))
            self.stdscr.addstr(3, 14, letter, color | curses.A_BOLD)
            
            # Confidence bar on the same line
            conf_label = "Confidence: "
            self.stdscr.addstr(3, 20, conf_label, curses.color_pair(5))
            
            bar_width = 30  # Fixed width to save space
            filled = int(bar_width * confidence)
            
            # Draw the bar
            self.stdscr.addstr(3, 20 + len(conf_label), "[" + "#" * filled + "-" * (bar_width - filled) + "]", color)
            
            # Confidence percentage
            conf_pct = f"{int(confidence * 100)}%"
            self.stdscr.addstr(3, 20 + len(conf_label) + bar_width + 2, conf_pct, color)
            
            # Recent predictions - more compact display
            self.stdscr.addstr(5, 2, "Recent:", curses.color_pair(4))
            
            # Display recent predictions horizontally to save space
            recent_x = 10
            for i, (idx, conf) in enumerate(zip(self.recent_predictions[-5:], self.recent_confidences[-5:])):
                if i >= 5:  # Show at most 5 recent predictions
                    break
                    
                l = self.class_names[idx]
                
                # Choose color based on confidence
                if conf >= 0.7:
                    color = curses.color_pair(1)  # Green
                elif conf >= 0.4:
                    color = curses.color_pair(2)  # Yellow
                else:
                    color = curses.color_pair(3)  # Red
                
                # Display in compact format
                self.stdscr.addstr(5, recent_x, f"{l}({conf:.2f})", color)
                recent_x += len(l) + 8  # Space for next prediction
            
            # Sensor data status
            buffer_sizes = [len(buffer) for buffer in self.sensor_buffers]
            buffer_status = f"Sensors: {buffer_sizes[0]}/{buffer_sizes[1]}/{buffer_sizes[2]}/{buffer_sizes[3]}/{buffer_sizes[4]} of {self.window_size}"
            self.stdscr.addstr(6, 2, buffer_status, curses.color_pair(5))
            
            # Instructions at the bottom
            self.stdscr.addstr(height-1, 2, "Press 'q' to quit", curses.color_pair(5))
            
            # Refresh the screen
            self.stdscr.refresh()
            
        except Exception as e:
            # If there's an error updating the display, fall back to text mode
            self.stdscr = None
            print(f"Error updating display: {e}")
            print(f"Falling back to text mode")

    def process_data_loop(self):
        """Main loop for processing data and making predictions"""
        self.log("Processing thread started.")
        last_display_update = 0
        update_interval = 0.2  # Update display every 200ms even if prediction hasn't changed
        
        while self.running:
            try:
                # Always update the display periodically if we have predictions
                current_time = time.time()
                if self.recent_predictions and current_time - last_display_update > update_interval:
                    last_idx = self.recent_predictions[-1] if self.recent_predictions else 0
                    last_conf = self.recent_confidences[-1] if self.recent_confidences else 0
                    self.update_display(last_idx, last_conf)
                    last_display_update = current_time
                
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

                # Store recent predictions
                self.recent_predictions.append(pred_idx)
                self.recent_confidences.append(confidence)
                if len(self.recent_predictions) > 10:
                    self.recent_predictions.pop(0)
                    self.recent_confidences.pop(0)

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
                        
                # Always update the display
                self.update_display(pred_idx, confidence)

                # Check for user input if we're using curses
                if self.stdscr:
                    try:
                        key = self.stdscr.getch()
                        if key == ord('q'):
                            self.running = False
                            break
                    except:
                        pass

            except Exception as e:
                if self.running:
                    self.log(f"Error in processing loop: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1) # Avoid tight loop on error
        self.log("Processing thread finished.")

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

        self.log("Real-time classification starting. Press Ctrl+C to stop.")

        try:
            # Monitor the C process
            while self.running:
                # Check if the C process is still running
                if self.process.poll() is not None:
                    exit_code = self.process.returncode
                    self.log(f"C process exited unexpectedly. Stopping...")
                    self.log(f"C process exit code: {exit_code}")
                    self.running = False
                    self.stop()
                    break
                
                # Sleep to avoid tight loop
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.log("\nKeyboard interrupt received. Stopping...")
            self.running = False
            self.stop()
        except Exception as e:
            self.log(f"Error in main loop: {e}")
            self.running = False
            self.stop()

    def _signal_handler(self, sig, frame):
        self.log(f"\nReceived signal {sig}. Stopping...")
        self.running = False
        self.stop()

    def stop(self):
        """Stop all threads and processes gracefully"""
        self.log("Stopping real-time classification...")
        self.running = False
            
        # Terminate the C process if it's still running
        if hasattr(self, 'process') and self.process.poll() is None:
            self.log("Terminating C process...")
            # Send SIGTERM to the process group
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception as e:
                self.log(f"Error sending SIGTERM to process group: {e}")
                
            # Fallback: terminate the process directly
            self.process.terminate() 
            try:
                # Wait for a short period for graceful termination
                self.process.wait(timeout=2.0)
                self.log("C process terminated gracefully.")
            except subprocess.TimeoutExpired:
                self.log("C process did not terminate gracefully after SIGTERM, sending SIGKILL...")
                self.process.kill() # Force kill if needed
                try:
                     self.process.wait(timeout=1.0) # Wait briefly for kill
                     self.log("C process killed.")
                except subprocess.TimeoutExpired:
                     self.log("Warning: C process did not respond to SIGKILL.")
            except Exception as e:
                 self.log(f"Error during C process termination: {e}")
        elif hasattr(self, 'process'):
             pass # Process already finished

        # Wait for threads to finish
        self.log("Waiting for threads to finish...")
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
             except Exception as e:
                  self.log(f"Error joining thread {thread.name}: {e}")

        self.log("Real-time classification stopped.")
        # Use os._exit(0) for a more immediate exit in case of lingering threads
        # sys.exit(0) # Use standard exit
        os._exit(0) # Force exit if necessary

def main(stdscr=None):
    parser = argparse.ArgumentParser(description="Real-time ASL classification from MPU6050 sensors")
    parser.add_argument('--model', type=str, default="best_rawsignal_cnn.pth",
                        help="Path to model .pth file (JIT version '<model_name>_jit.pt' will be checked if --use-jit is set)")
    parser.add_argument('--window', type=int, default=91,
                        help="Window size (samples)")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Confidence threshold for predictions")
    parser.add_argument('--i2c', type=str, default="/dev/i2c-1",
                        help="I2C device path")
    parser.add_argument('--use-jit', action='store_true',
                        help="Use TorchScript JIT model if available (looks for <model_name>_jit.pt)")
    parser.add_argument('--no-gui', action='store_true',
                        help="Disable curses GUI and use text mode only")
    args = parser.parse_args()

    # List all files in the current directory to help debug model loading issues
    print("\nListing files in current directory:")
    for file in os.listdir("."):
        if file.endswith(".pth") or file.endswith(".pt"):
            print(f"  Found model file: {file}")
        elif os.path.isdir(file):
            print(f"  Directory: {file}")
    print()

    # Check if the model file exists
    model_exists = os.path.exists(args.model)
    jit_model_path = args.model.replace('.pth', '_jit.pt')
    jit_model_exists = os.path.exists(jit_model_path)

    # If model doesn't exist in current directory, try looking in common subdirectories
    if not model_exists and not jit_model_exists:
        possible_dirs = ["models", "weights", "checkpoints"]
        for dir_name in possible_dirs:
            if os.path.isdir(dir_name):
                print(f"Looking for models in {dir_name}/")
                for file in os.listdir(dir_name):
                    if file.endswith(".pth") or file.endswith(".pt"):
                        print(f"  Found model in subdirectory: {dir_name}/{file}")
                        if file == os.path.basename(args.model):
                            args.model = os.path.join(dir_name, file)
                            model_exists = True
                            print(f"Using model at: {args.model}")
                        elif file == os.path.basename(jit_model_path):
                            jit_model_path = os.path.join(dir_name, file)
                            jit_model_exists = True
                            print(f"Using JIT model at: {jit_model_path}")

    # Final check if model exists
    if not model_exists and not (args.use_jit and jit_model_exists):
        print("\nERROR: Model file not found!")
        print(f"Checked for: {args.model}")
        if args.use_jit:
            print(f"Checked for JIT model: {jit_model_path}")
        
        # Look for any model file as a fallback
        fallback_models = []
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".pth") or file.endswith(".pt"):
                    fallback_models.append(os.path.join(root, file))
        
        if fallback_models:
            print("\nFound these model files that could be used instead:")
            for model in fallback_models:
                print(f"  {model}")
            # Use the first one as fallback
            args.model = fallback_models[0]
            print(f"\nUsing {args.model} as fallback model")
            model_exists = True
        else:
            print("No model files found in this directory or subdirectories.")
            sys.exit(1)
    
    if args.use_jit and not jit_model_exists and model_exists:
        print(f"Warning: --use-jit specified, but JIT model '{jit_model_path}' not found. Will use regular model '{args.model}'.")

    # Create and start the processor
    processor = SensorDataProcessor(
        model_path=args.model,
        window_size=args.window,
        confidence_threshold=args.threshold,
        use_jit=args.use_jit,
        stdscr=None if args.no_gui else stdscr
    )

    processor.start(i2c_device=args.i2c)

if __name__ == "__main__":
    # Check if we should use curses or not
    if '--no-gui' in sys.argv:
        main()
    else:
        try:
            curses.wrapper(main)
        except Exception as e:
            print(f"Error initializing curses: {e}")
            print("Falling back to text mode")
            sys.argv.append('--no-gui')
            main()
