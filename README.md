# Real-time ASL Classification with MPU6050 Sensors and PyTorch

This project implements a real-time American Sign Language (ASL) classification system using MPU6050 inertial sensors and a PyTorch deep learning model. The system is designed to run on a Jetson Nano with Docker support.

## System Overview

The system captures motion data from multiple MPU6050 sensors connected via a TCA9548A I2C multiplexer, processes this data in real-time, and classifies the hand gestures using a trained CNN model. The classification results are output to the console with confidence scores.

![System Architecture](https://i.imgur.com/PLACEHOLDER.png)

## Components

### Hardware Requirements

- NVIDIA Jetson Nano
- Multiple MPU6050 sensors (up to 5)
- TCA9548A I2C multiplexer
- I2C connections to the Jetson Nano

### Software Components

1. **`NEW_C.c`**: C program that interfaces with the MPU6050 sensors via I2C
   - Initializes and reads data from multiple MPU6050 sensors
   - Outputs sensor data in the format: `sensor_idx gx gy gz ax ay az`
   - Handles I2C communication and multiplexer channel selection
   - Provides real-time streaming of sensor data

2. **`best_rawsignal_cnn.pth`**: Pre-trained PyTorch model for ASL classification
   - Trained on raw signal data from MPU6050 sensors
   - Classifies 27 different ASL signs (A-Z plus Rest)
   - Uses a 3D CNN architecture optimized for temporal sensor data

3. **`realtime_classifier.py`**: Python script for real-time classification
   - Reads sensor data from the C program
   - Processes data in sliding windows
   - Uses data in the format provided by the C program (`gx, gy, gz, ax, ay, az`)
   - Performs inference using the PyTorch model
   - Outputs predictions with confidence scores

4. **`convert_to_torchscript.py`**: Utility to convert the PyTorch model to TorchScript
   - Creates an optimized version of the model for deployment
   - Verifies the converted model works correctly
   - Improves inference performance in the Docker container

5. **`run_asl_classifier.sh`**: Launcher script for the Docker container
   - Compiles the C code if needed
   - Converts the model to TorchScript if needed
   - Finds the Jetson Docker run script
   - Launches the Docker container with necessary mounts and devices
   - **Now correctly runs commands inside the container using `/bin/bash -c`, fixing issues with `cd` not found errors**

6. **`run_direct.sh`**: Direct launcher script for the Docker container (recommended)
   - Compiles the C code if needed
   - Directly runs the Docker container with full GPU access
   - Mounts all necessary devices (I2C, display, video)
   - Includes all Jetson-specific system mounts
   - More reliable than using the wrapper script

6. **`Dockerfile`**: Definition for the Docker container (optional)
   - Based on the NVIDIA L4T PyTorch container
   - Includes all necessary dependencies
   - Note: The system primarily uses the existing dusty-nv container

## Data Flow

1. The C program (`mpu6050_reader` compiled from `NEW_C.c`) reads raw sensor data from multiple MPU6050 sensors
2. The sensor data is streamed as text output in the format: `sensor_idx gx gy gz ax ay az`
3. The Python classifier (`realtime_classifier.py`) captures this output and buffers it
4. When enough data is collected (window size), it's processed and fed to the neural network
5. The model performs inference and outputs the predicted ASL sign with a confidence score
6. The process repeats continuously for real-time classification

## Installation and Setup

### Prerequisites

- Jetson Nano with JetPack 4.6 or later
- Docker installed (dusty-nv containers)
- I2C enabled on the Jetson Nano
- Python 3.6+ with PyTorch 1.10+

### Hardware Setup

1. Connect the MPU6050 sensors to the TCA9548A I2C multiplexer
2. Connect the multiplexer to the Jetson Nano's I2C bus
3. Power the sensors and multiplexer appropriately

### Software Setup

1. Clone this repository to your Jetson Nano:
   ```bash
   git clone https://github.com/DrFarrukh/ASL-classification-C.git
   cd ASL-classification-C
   ```

2. Compile the C code:
   ```bash
   gcc -o mpu6050_reader NEW_C.c -lm
   chmod +x mpu6050_reader
   ```

3. Convert the PyTorch model to TorchScript (optional but recommended):
   ```bash
   python3 convert_to_torchscript.py
   ```

4. Make the launcher script executable:
   ```bash
   chmod +x run_asl_classifier.sh
   ```

## Usage

1. Run the direct launcher script (recommended):
   ```bash
   ./run_direct.sh
   ```

2. The script will:
   - Compile the C code if needed
   - Set up display forwarding if available
   - Detect and mount video and I2C devices
   - Create the necessary system mounts for GPU access
   - Run the Docker container with full GPU access
   - Execute the realtime classifier inside the container

3. Alternatively, you can use the original launcher script that uses the jetson-inference Docker run script:
   ```bash
   ./run_asl_classifier.sh
   ```

3. Inside the Docker container, the classifier will:
   - Read data from the MPU6050 sensors
   - Process the data in sliding windows
   - Perform inference using the PyTorch model
   - Output predictions when they meet the confidence threshold

4. To stop the classifier, press `Ctrl+C` in the terminal.

## Customization

### Adjusting Parameters

You can modify the following parameters in the `realtime_classifier.py` script:

- `--window`: Window size (number of samples) for classification (default: 91)
- `--threshold`: Confidence threshold for predictions (default: 0.6)
- `--i2c`: I2C device path (default: /dev/i2c-1)
- `--use-jit`: Whether to use the TorchScript model (recommended)

Example:
```bash
python3 realtime_classifier.py --window 100 --threshold 0.7 --use-jit
```

### Modifying the Model

If you want to use a different model:

1. Replace `best_rawsignal_cnn.pth` with your model
2. Update the model architecture in `realtime_classifier.py` if needed
3. Update the `convert_to_torchscript.py` script to match your model architecture
4. Run the conversion again: `python3 convert_to_torchscript.py`

## Troubleshooting

### I2C Issues

- Ensure I2C is enabled on your Jetson Nano
- Check connections between sensors, multiplexer, and Jetson
- Verify I2C addresses in the C code match your hardware
- Run `i2cdetect -y -r 1` to check for connected I2C devices

### Docker Issues

- Ensure Docker is properly installed on your Jetson Nano
- Verify the path to the Docker run script in `run_asl_classifier.sh`
- Check that the Docker container has access to the I2C devices
- **If you previously saw an error like `exec: "cd": executable file not found in $PATH`, this has been fixed in the latest version of `run_asl_classifier.sh` by running commands using `/bin/bash -c` inside the container.**
- **If you encountered I2C device access issues (`Error opening I2C device /dev/i2c-1: No such file or directory`), this has been fixed by explicitly mapping I2C devices into the Docker container.**

### Classification Issues

- Increase the window size for more stable predictions
- Adjust the confidence threshold based on your needs
- Ensure the sensors are properly calibrated and positioned

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the PyTorch deep learning framework
- The MPU6050 sensor interface is based on the I2C protocol
- Special thanks to the NVIDIA Jetson community for Docker support

### GUI

- **Real-time Processing**: Performs real-time signal processing (windowing, normalization) and feature extraction (scalograms if using the scalogram model).
- **Model Inference**: Loads a pre-trained PyTorch model (`.pth` or `.pt`) for either raw signals or scalograms and performs inference on incoming data windows.
- **GUI**: Provides a user-friendly Tkinter interface displaying:
  - The currently predicted ASL letter.
  - Confidence score for the prediction.
  - A bar chart of probabilities for all classes.
  - A history log of predictions.
  - **A dedicated canvas for visualizing the generated scalogram grid (when using `--model-type scalogram`), using the same standardized method as other analysis scripts.**
- **Sensor Reading**: Communicates with the C program (`NEW_C`) via standard output to get sensor readings.
- **Configuration**: Supports command-line arguments for model path, model type, **window size (default 91)**, confidence threshold, and I2C device.

### Visualization & Testing Tools

- **`basic_scalogram.py`**: A simplified Tkinter-based viewer focusing purely on visualizing the 5-sensor data as a scalogram grid. Imports `pywt` and `PIL` if available.
- **`simple_viewer.py`**: A minimal Tkinter viewer that prints raw sensor values and provides a basic line plot visualization without complex processing or dependencies.
- **`scalogram_viewer.py`**: A standalone Tkinter tool dedicated to visualizing scalograms generated from live sensor data or test data. It shows both the full 2x5 grid and individual sensor scalograms.
- **`csv_scalogram_test.py`**: A script to load sensor data from a CSV file and visualize it using the `basic_scalogram.py` viewer logic.
- **`csv_scalogram_exact.py`**: **A script to load sensor data from a CSV file and visualize the scalogram grid using the *exact* same generation method (`rgb_scalogram_3axis`, `make_scalogram_grid`) employed by `stream_real_data.py` and the GUI (`realtime_classifier_gui.py` when `--model-type scalogram`). Useful for verifying data format and visualization consistency.**

### Running the GUI Classifier

1.  **Ensure the C program is running** and printing data to stdout.
2.  Run the Python script, specifying the model path and type:

    ```bash
    # Example for Raw Signal Model
    python3 realtime_classifier_gui.py --model best_rawsignal_cnn.pth --model-type raw --window 91

    # Example for Scalogram Model (requires a scalogram-trained model)
    python3 realtime_classifier_gui.py --model best_scalogram_cnn.pth --model-type scalogram --window 91
    ```

    *   Replace `best_rawsignal_cnn.pth` or `best_scalogram_cnn.pth` with the actual path to your trained model.
    *   The `--window` argument should generally match the window size used during model training (default is now 91).
    *   Use `--model-type raw` for models trained on raw signals (like `RawSignalCNN`).
    *   Use `--model-type scalogram` for models trained on scalogram images (like `SimpleJetsonCNN`).

### Running Offline Analysis (`stream_real_data.py`)

This script simulates real-time processing using pre-recorded data.

```bash
python3 stream_real_data.py --data path/to/your_data.csv --model path/to/your_model.pth --mode [raw|scalogram] --window 91
```

### Running Visualization Tools

- **Simple Viewer**:
  ```bash
  # Pipe C program output to the viewer
  ./NEW_C | python3 simple_viewer.py
  # Or run in test mode (generates synthetic data)
  python3 simple_viewer.py --test
  ```
- **Basic Scalogram Viewer**:
  ```bash
  ./NEW_C | python3 basic_scalogram.py --window 91
  python3 basic_scalogram.py --test --window 91
  ```
- **Dedicated Scalogram Viewer**:
  ```bash
  ./NEW_C | python3 scalogram_viewer.py --window 91
  python3 scalogram_viewer.py --test --window 91
  ```
- **CSV Scalogram Test (Basic)**:
  ```bash
  python3 csv_scalogram_test.py path/to/data.csv --window 91
  ```
- **CSV Scalogram Test (Exact Method)**:
  ```bash
  python3 csv_scalogram_exact.py path/to/data.csv --window 91
  # Optionally save images
  python3 csv_scalogram_exact.py path/to/data.csv --window 91 --save
  ```
