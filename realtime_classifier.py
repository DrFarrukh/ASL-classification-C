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
    def __init__(self, model_path, window_size=91, confidence_threshold=0.6, use_jit=True):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.num_classes = 27
        self.class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','Rest','S','T','U','V','W','X','Y','Z']

        # Initialize data buffers for each sensor
        self.sensor_buffers = [deque(maxlen=window_size) for _ in range(5)]
        self.data_queue = queue.Queue()
        self.running = True
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_report_time = 0 # Initialize last report time

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
                sys.exit(1)

        print(f"Starting C process: {c_executable} {i2c_device}")
        try:
            self.process = subprocess.Popen(
                [c_executable, i2c_device],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, # Capture stderr
                bufsize=1, # Line buffered
                universal_newlines=True, # Decode as text
                # Set process group ID to allow killing the whole group later if needed
                # preexec_fn=os.setsid 
            )
        except OSError as e:
            print(f"Error starting C process: {e}. Is the path correct and does it have execute permissions?")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error starting C process: {e}")
            sys.exit(1)

        print(f"Started sensor data collection process (PID: {self.process.pid})")

        # Start thread to read from process stdout
        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

        # Start thread to read from process stderr
        self.stderr_thread = threading.Thread(target=self._read_process_stderr, daemon=True)
        self.stderr_thread.start()

    def _read_process_output(self):
        """Read data from the C process stdout"""
        print("Stdout reader thread started.")
        # Ensure process and stdout exist before entering loop
        if not hasattr(self, 'process') or not self.process.stdout:
             print("Error: C process or stdout not available for reading.")
             self.running = False # Signal main loop to stop
             return

        while self.running:
            try:
                # Check if process has exited before reading
                if self.process.poll() is not None:
                    # print("C process exited while reading stdout.")
                    break

                line = self.process.stdout.readline()
                if not line:
                    # Check again if process exited after readline returned empty
                    if self.process.poll() is not None:
                        # print("C process stdout stream ended gracefully.")
                        break
                    # If process is still running, readline might timeout or encounter EOF temporarily
                    time.sleep(0.01) # Small sleep to avoid busy-waiting
                    continue

                line = line.strip()
                if not line:
                    continue

                # Debug print raw line
                # print(f"Raw C Output: {line}")

                # Parse the line: sensor_index ax ay az gx gy gz
                parts = line.split()
                if len(parts) != 7:
                    # print(f"Skipping invalid line (expected 7 parts, got {len(parts)}): {line}")
                    continue

                # Use try-except for robust parsing
                try:
                    sensor_idx = int(parts[0])
                    # Validate sensor index range
                    if not (0 <= sensor_idx < 5):
                        # print(f"Skipping invalid sensor index ({sensor_idx}): {line}")
                        continue

                    # Extract sensor values in the order they appear in the C output
                    # C Output: sensor_idx gx gy gz ax ay az
                    gx, gy, gz, ax, ay, az = map(float, parts[1:])

                    # Create the data point - already in the correct order for the model
                    # Model Input Order: gx, gy, gz, ax, ay, az
                    data_point = [gx, gy, gz, ax, ay, az]

                except ValueError as e:
                    # print(f"Error parsing numeric values in line '{line}': {e}")
                    continue # Skip this line

                # Add to the appropriate sensor buffer
                self.sensor_buffers[sensor_idx].append(data_point)

                # Check if we have enough data for inference (moved outside parsing try-except)
                self._check_and_process_data()

            except Exception as e:
                # Catch unexpected errors during the read/process loop
                if self.running: # Avoid printing errors during shutdown
                     print(f"Unexpected error in stdout reader thread: {e}")
                     import traceback
                     traceback.print_exc()
                time.sleep(0.1) # Avoid tight loop on continuous error

        print("Stdout reader thread finished.")

    def _read_process_stderr(self):
        """Read data from the C process stderr"""
        print("Stderr reader thread started.")
        if not hasattr(self, 'process') or not self.process.stderr:
             print("Error: C process or stderr not available for reading.")
             return
             
        while self.running:
            try:
                if self.process.poll() is not None:
                    # Read remaining lines after process exit
                    for line in self.process.stderr.readlines():
                        print(f"[C Stderr - Final]: {line.strip()}")
                    break
                    
                line = self.process.stderr.readline()
                if not line:
                    if self.process.poll() is not None:
                        # print("C process stderr stream ended gracefully.")
                        break
                    time.sleep(0.01)
                    continue
                print(f"[C Stderr]: {line.strip()}")
            except Exception as e:
                if self.running:
                    print(f"Error reading C process stderr: {e}")
                time.sleep(0.1)
        print("Stderr reader thread finished.")

    def _check_and_process_data(self):
        """Check if all buffers are full and queue data for prediction."""
        # This function might be called frequently, so keep it efficient
        # Check if ALL sensor buffers have reached the required window size
        # Using min() is slightly faster than all() for potentially large number of sensors
        if len(self.sensor_buffers) > 0 and min(len(buf) for buf in self.sensor_buffers) >= self.window_size:
            # Create the window data array only when all buffers are ready
            # Shape: (5, window_size, 6)
            # We take the most recent 'window_size' items from each deque
            window_data = np.array([list(buffer) for buffer in self.sensor_buffers]) # list() converts deque slice

            # Add to queue for processing
            self.data_queue.put(window_data)

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

            except Exception as e:
                if self.running:
                    print(f"Error in processing loop: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1) # Avoid tight loop on error
        print("Processing thread finished.")

    def start(self, i2c_device="/dev/i2c-1"):
        """Start the data collection and processing"""
        self.last_report_time = time.time() # Initialize report time

        # Start the C process
        self.start_c_process(i2c_device)

        # Give the C process a moment to start up and initialize sensors
        # This might be necessary if the C code takes time before outputting data
        print("Waiting for C process to initialize...")
        time.sleep(2.0)

        # Start the processing thread
        self.processor_thread = threading.Thread(target=self.process_data_loop, daemon=True)
        self.processor_thread.start()

        print("Real-time classification starting. Press Ctrl+C to stop.")

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Keep the main thread alive, checking if threads/process are running
        try:
            while self.running:
                # Check if C process terminated unexpectedly
                if hasattr(self, 'process') and self.process.poll() is not None:
                    print("C process exited unexpectedly. Stopping...")
                    exit_code = self.process.poll()
                    print(f"C process exit code: {exit_code}")
                    # Read remaining stderr after exit
                    if self.process.stderr:
                         for line in self.process.stderr.readlines():
                              print(f"[C Stderr - Final]: {line.strip()}")
                    self.stop()
                    break
                
                # Check if threads died unexpectedly
                if hasattr(self, 'reader_thread') and not self.reader_thread.is_alive() and self.running:
                    print("Stdout reader thread died unexpectedly. Stopping...")
                    self.stop()
                    break
                if hasattr(self, 'stderr_thread') and not self.stderr_thread.is_alive() and self.running:
                    print("Stderr reader thread died unexpectedly. Stopping...")
                    self.stop()
                    break
                if hasattr(self, 'processor_thread') and not self.processor_thread.is_alive() and self.running:
                     print("Processor thread died unexpectedly. Stopping...")
                     self.stop()
                     break
                     
                time.sleep(0.5) # Main loop check interval
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received.")
            self.stop()
        finally:
            # Ensure stop is called if loop exits for any other reason
            if self.running:
                print("Main loop finished unexpectedly. Stopping...")
                self.stop()

    def _signal_handler(self, sig, frame):
        # Check if already stopping to prevent multiple calls
        if not self.running:
             return
        print(f"\nReceived signal {sig}. Initiating shutdown...")
        self.stop()

    def stop(self):
        """Stop all threads and processes gracefully"""
        if not self.running:
            # print("Already stopping.")
            return # Already stopping/stopped

        print("Stopping real-time classification...")
        self.running = False # Signal threads to stop

        # Terminate the C process
        if hasattr(self, 'process') and self.process.poll() is None:
            print("Terminating C process...")
            # Try SIGTERM first
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
        

    processor = SensorDataProcessor(
        model_path=args.model,
        window_size=args.window,
        confidence_threshold=args.threshold,
        use_jit=args.use_jit
    )

    processor.start(i2c_device=args.i2c)

if __name__ == "__main__":
    main()
