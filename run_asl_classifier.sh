#!/bin/bash

# Exit on error
set -e

# Directory where the ASL project is located
ASL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ASL_DIR"

# Check if the C executable exists, if not compile it
if [ ! -f "mpu6050_reader" ]; then
    echo "Compiling C executable..."
    gcc -o mpu6050_reader NEW_C.c -lm
    echo "Compilation complete."
fi

# Create a TorchScript version of the model if it doesn't exist
if [ ! -f "best_rawsignal_cnn_jit.pt" ]; then
    echo "Converting PyTorch model to TorchScript..."
    python3 convert_to_torchscript.py
    echo "Conversion complete."
fi

# Set execute permissions
chmod +x mpu6050_reader

# Determine the path to the docker run.sh script
# Try common locations for the jetson-inference repository
POTENTIAL_PATHS=(
    "/opt/jetson-inference/docker/run.sh"
    "/usr/local/src/jetson-inference/docker/run.sh"
    "$HOME/jetson-inference/docker/run.sh"
    "$HOME/projects/jetson-inference/docker/run.sh"
    "$HOME/workspace/jetson-inference/docker/run.sh"
)

DOCKER_SCRIPT_PATH=""
for path in "${POTENTIAL_PATHS[@]}"; do
    if [ -f "$path" ]; then
        DOCKER_SCRIPT_PATH="$path"
        break
    fi
done

# If not found in common locations, ask the user
if [ -z "$DOCKER_SCRIPT_PATH" ]; then
    echo "Could not find docker/run.sh in common locations."
    echo "Please enter the full path to your jetson-inference docker/run.sh script:"
    read -p "> " DOCKER_SCRIPT_PATH
    
    if [ ! -f "$DOCKER_SCRIPT_PATH" ]; then
        echo "Error: Docker run script not found at $DOCKER_SCRIPT_PATH"
        exit 1
    fi
fi

echo "Using Docker run script at: $DOCKER_SCRIPT_PATH"

# Run the Docker container using the existing script
# We need to:
# 1. Mount our ASL directory
# 2. Add access to I2C devices
# 3. Run our realtime_classifier.py script

echo "Starting ASL classifier in Docker container..."
# Change to the jetson-inference directory first
DOCKER_DIR=$(dirname "$DOCKER_SCRIPT_PATH")
JETSON_INFERENCE_DIR=$(dirname "$DOCKER_DIR")
cd "$JETSON_INFERENCE_DIR"

# Since the dusty-nv script doesn't properly pass I2C devices, we'll use our direct approach
# instead of relying on the dusty-nv script

echo "Running Docker container directly with I2C device access..."

# Create temporary file for Jetson model info
cat /proc/device-tree/model > /tmp/nv_jetson_model 2>/dev/null || echo "Not a Jetson device" > /tmp/nv_jetson_model

# Run the Docker container with full GPU access and system mounts
sudo docker run --runtime nvidia -it --rm \
    --network host \
    --device /dev/i2c-0 \
    --device /dev/i2c-1 \
    --device /dev/i2c-2 \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v /etc/nv_tegra_release:/etc/nv_tegra_release \
    -v /tmp/nv_jetson_model:/tmp/nv_jetson_model \
    -v /var/run/dbus:/var/run/dbus \
    -v /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    -v "$ASL_DIR:/asl" \
    -w /asl \
    dustynv/jetson-inference:r32.7.1 \
    bash -c "cd /asl && python3 realtime_classifier_simple.py --use-jit --threshold 0.6"

# Return to the original directory
cd "$ASL_DIR"

echo "ASL classifier Docker container exited."
