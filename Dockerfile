FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    i2c-tools \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    pywavelets \
    matplotlib \
    pillow \
    pyttsx3

# Create working directory
WORKDIR /app

# Copy only the necessary files
COPY best_rawsignal_cnn.pth /app/
COPY realtime_classifier.py /app/
COPY mpu6050_reader /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "realtime_classifier.py", "--use-jit"]
