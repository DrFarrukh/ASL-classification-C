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

# Set execute permissions
chmod +x mpu6050_reader

echo "Starting ASL classifier in Docker container..."

# Check for display
if [ -n "$DISPLAY" ]; then
    sudo xhost +si:localuser:root
    DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix"
fi

# Check for V4L2 devices
V4L2_DEVICES=""
for i in {0..9}; do
    if [ -a "/dev/video$i" ]; then
        V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i"
    fi
done

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
    $DISPLAY_DEVICE $V4L2_DEVICES \
    -v "$ASL_DIR:/asl" \
    -w /asl \
    dustynv/jetson-inference:r32.7.1 \
    bash -c "cd /asl && pip3 install matplotlib pywavelets && python3 realtime_classifier_visual.py --use-jit --threshold 0.3"

echo "ASL classifier Docker container exited."
