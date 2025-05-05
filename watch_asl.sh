#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASL_DIR="$SCRIPT_DIR"

echo "Starting ASL classifier with watch display..."

# Create temporary file for Jetson model info
cat /proc/device-tree/model > /tmp/nv_jetson_model 2>/dev/null || echo "Not a Jetson device" > /tmp/nv_jetson_model

# Output file for JSON data
OUTPUT_FILE="/tmp/asl_output.json"

# Run the classifier in the background
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
    -v "/tmp:/tmp" \
    -w /asl \
    dustynv/jetson-inference:r32.7.1 \
    bash -c "cd /asl && python3 asl_text_output.py --use-jit --threshold 0.3 --output-file $OUTPUT_FILE --slow" &

# Store the Docker process ID
DOCKER_PID=$!

# Wait a moment for the classifier to start
sleep 2

# Function to display the ASL output in a nice format
display_asl_output() {
    if [ -f "$OUTPUT_FILE" ]; then
        # Extract data from JSON file
        PREDICTION=$(jq -r '.prediction' "$OUTPUT_FILE")
        CONFIDENCE=$(jq -r '.confidence' "$OUTPUT_FILE")
        BUFFERS=$(jq -r '.buffers | join("/")' "$OUTPUT_FILE")
        TARGET=$(jq -r '.target' "$OUTPUT_FILE")
        
        # Format confidence as percentage
        CONF_PCT=$(echo "$CONFIDENCE * 100" | bc -l | xargs printf "%.1f")
        
        # Create a bar for confidence
        BAR_LEN=40
        FILLED_LEN=$(echo "$CONFIDENCE * $BAR_LEN" | bc -l | xargs printf "%.0f")
        BAR=$(printf "%${FILLED_LEN}s" | tr ' ' '#')
        EMPTY_LEN=$((BAR_LEN - FILLED_LEN))
        if [ $EMPTY_LEN -gt 0 ]; then
            EMPTY=$(printf "%${EMPTY_LEN}s" | tr ' ' '-')
            BAR="$BAR$EMPTY"
        fi
        
        # Display header
        echo "========================================================"
        echo "             ASL CLASSIFIER - $(date +%H:%M:%S)"
        echo "========================================================"
        echo ""
        echo "  CURRENT PREDICTION: $PREDICTION"
        echo "  Confidence: [$BAR] $CONF_PCT%"
        echo ""
        echo "  Sensor buffers: $BUFFERS of $TARGET"
        echo ""
        
        # Get recent predictions
        RECENT=$(jq -r '.recent | map(.letter + "(" + (.confidence * 100 | floor | tostring) + "%)") | join(" ")' "$OUTPUT_FILE")
        if [ -n "$RECENT" ]; then
            echo "  Recent: $RECENT"
        fi
        
        echo ""
        echo "  Press Ctrl+C to stop"
        echo "========================================================"
    else
        echo "Waiting for ASL classifier to start..."
    fi
}

# Use watch to display the output at a slower rate
watch -n 1 -t "$(declare -f display_asl_output); display_asl_output"

# When watch is terminated, kill the Docker container
kill $DOCKER_PID 2>/dev/null
wait $DOCKER_PID 2>/dev/null

echo "ASL classifier with watch display exited."
