#!/bin/bash

# Exit on error
set -e

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

# Define the container name
CONTAINER_NAME="asl_classifier"

# Check if container already exists
if [ "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
fi

# Run the Docker container with necessary permissions
echo "Starting Docker container..."
docker run --runtime nvidia \
    --name $CONTAINER_NAME \
    -it \
    --network host \
    --device /dev/i2c-1 \
    -v $(pwd):/app \
    -w /app \
    dustynv/l4t-pytorch:r32.7.1-pth1.10-py3 \
    python3 realtime_classifier.py --use-jit

echo "Container started."
