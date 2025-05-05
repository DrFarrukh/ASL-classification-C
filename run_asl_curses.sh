#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASL_DIR="$SCRIPT_DIR"

echo "Starting ASL classifier with curses GUI display..."

# Create temporary file for Jetson model info
cat /proc/device-tree/model > /tmp/nv_jetson_model 2>/dev/null || echo "Not a Jetson device" > /tmp/nv_jetson_model

# Set up X11 display for GUI
DISPLAY_DEVICE=""
if [ -n "$DISPLAY" ]; then
    XAUTH=/tmp/.docker.xauth
    touch $XAUTH
    xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    DISPLAY_DEVICE="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH"
fi

# Run the Docker container with I2C device access
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
    $DISPLAY_DEVICE \
    -v "$ASL_DIR:/asl" \
    -w /asl \
    -e TERM=xterm \
    dustynv/jetson-inference:r32.7.1 \
    bash -c "cd /asl && python3 asl_curses_display.py --use-jit --threshold 0.3"

echo "ASL classifier with curses GUI display exited."
