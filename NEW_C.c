/**
 * MPU6050 Reader for Jetson Nano
 * 
 * This program reads data from multiple MPU6050 sensors connected via a TCA9548A I2C multiplexer
 * and outputs the accelerometer and gyroscope values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <errno.h>
#include <string.h>
#include <signal.h>
#include <time.h>

// I2C addresses
#define TCA9548A_ADDR 0x70  // TCA9548A I2C multiplexer address
#define MPU6050_ADDR  0x68  // MPU6050 I2C address

// MPU6050 registers
#define MPU6050_REG_PWR_MGMT_1    0x6B
#define MPU6050_REG_ACCEL_XOUT_H  0x3B
#define MPU6050_REG_WHO_AM_I      0x75

// Number of sensors
#define NUM_SENSORS 5

// Debug mode
#define DEBUG 1

// Global variables
int g_i2c_fd = -1;
volatile int g_running = 1;

// Function prototypes
int open_i2c_device(const char *device);
int tca_select(int i2c_fd, uint8_t channel);
int mpu6050_initialize(int i2c_fd);
int mpu6050_test_connection(int i2c_fd);
int mpu6050_get_motion_6(int i2c_fd, int16_t *ax, int16_t *ay, int16_t *az, int16_t *gx, int16_t *gy, int16_t *gz);
int i2c_write_byte(int i2c_fd, uint8_t dev_addr, uint8_t reg_addr, uint8_t data);
int i2c_read_byte(int i2c_fd, uint8_t dev_addr, uint8_t reg_addr, uint8_t *data);
int i2c_read_bytes(int i2c_fd, uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, size_t length);
void scan_i2c_bus(int i2c_fd);
void signal_handler(int sig);

int main(int argc, char *argv[]) {
    int16_t ax, ay, az, gx, gy, gz;
    const char *i2c_device = "/dev/i2c-1";  // Default I2C bus
    
    // Handle command line arguments
    if (argc > 1) {
        i2c_device = argv[1];
    }
    
    // Set up signal handler for graceful exit
    signal(SIGINT, signal_handler);
    
    printf("MPU6050 Reader for Jetson Nano\n");
    printf("Using I2C bus: %s\n", i2c_device);
    
    // Open I2C device
    g_i2c_fd = open_i2c_device(i2c_device);
    if (g_i2c_fd < 0) {
        fprintf(stderr, "Failed to open I2C device %s\n", i2c_device);
        return 1;
    }
    
    // Scan I2C bus for devices
    printf("Scanning I2C bus for devices...\n");
    scan_i2c_bus(g_i2c_fd);
    
    // Check if TCA9548A is present
    uint8_t data;
    if (i2c_read_byte(g_i2c_fd, TCA9548A_ADDR, 0, &data) < 0) {
        fprintf(stderr, "TCA9548A multiplexer not found at address 0x%02X\n", TCA9548A_ADDR);
        printf("Continuing without multiplexer (assuming direct connection to MPU6050)\n");
    } else {
        printf("TCA9548A multiplexer found at address 0x%02X\n", TCA9548A_ADDR);
    }
    
    // Initialize each sensor
    int active_sensors = 0;
    int sensor_status[NUM_SENSORS] = {0};
    
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        printf("Checking sensor %d...\n", i);
        
        // Try to select the channel, but continue even if it fails
        if (tca_select(g_i2c_fd, i) < 0) {
            printf("Failed to select channel %d on TCA9548A, but continuing anyway\n", i);
        }
        
        // Small delay after channel selection
        usleep(10000);  // 10ms
        
        if (mpu6050_initialize(g_i2c_fd) < 0) {
            printf("Failed to initialize MPU6050 on channel %d\n", i);
            sensor_status[i] = 0;
            continue;
        }
        
        if (!mpu6050_test_connection(g_i2c_fd)) {
            printf("MPU6050 on channel %d connection failed!\n", i);
            sensor_status[i] = 0;
        } else {
            printf("MPU6050 on channel %d connected successfully.\n", i);
            sensor_status[i] = 1;
            active_sensors++;
        }
        
        usleep(100000);  // 100ms delay
    }
    
    if (active_sensors == 0) {
        printf("No active sensors found. Trying direct connection to MPU6050...\n");
        
        if (mpu6050_initialize(g_i2c_fd) < 0) {
            printf("Failed to initialize MPU6050 with direct connection\n");
        } else if (!mpu6050_test_connection(g_i2c_fd)) {
            printf("MPU6050 direct connection failed!\n");
        } else {
            printf("MPU6050 direct connection successful.\n");
            // Set the first sensor as active
            sensor_status[0] = 2;  // 2 means direct connection
            active_sensors = 1;
        }
    }
    
    if (active_sensors == 0) {
        printf("No MPU6050 sensors found. Exiting.\n");
        close(g_i2c_fd);
        return 1;
    }
    
    printf("Found %d active sensor(s). Starting data collection...\n", active_sensors);
   

    while (g_running) {
        for (uint8_t i = 0; i < NUM_SENSORS; i++) {
            if (!sensor_status[i]) continue;  // Skip inactive sensors

            if (sensor_status[i] == 1) {  // Using multiplexer
                if (tca_select(g_i2c_fd, i) < 0) {
                    printf("Failed to select channel %d, skipping\n", i);
                    continue;
                }
                usleep(5000);  //  delay after channel selection
            }

            if (mpu6050_get_motion_6(g_i2c_fd, &ax, &ay, &az, &gx, &gy, &gz) >= 0) {
                // Output data to stdout: sensor_index gx gy gz ax ay az
                fprintf(stderr, "C: Writing sensor %d data to pipe\n", i);
                printf("%d %d %d %d %d %d %d\n", i, gx, gy, gz, ax, ay, az);
                fflush(stdout); // Ensure data is printed immediately
                
                // Write sensor data to CSV file
                // time_t now = time(NULL);
                // fprintf(fp, "%ld,%d,%d,%d,%d,%d,%d,%d\\n", now, i, ax, ay, az, gx, gy, gz);
                // fflush(fp); // Immediately flush data to CSV
            } else {
                printf("Failed to read data from sensor %d\n", i);
            }

            usleep(2500);  //  delay between readings
        }

        usleep(5000);  // additional delay to make it approximately 500ms total
    }

    // fclose(fp); // Close the file before exiting

    printf("Exiting...\n");
    close(g_i2c_fd);
    return 0;
}

// Signal handler for graceful exit
void signal_handler(int sig) {
    printf("\nReceived signal %d, exiting...\n", sig);
    g_running = 0;
}

// Open I2C device
int open_i2c_device(const char *device) {
    int fd = open(device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Error opening I2C device %s: %s\n", device, strerror(errno));
        return -1;
    }
    return fd;
}

// Scan I2C bus for devices
void scan_i2c_bus(int i2c_fd) {
    int found = 0;
    
    for (int addr = 0x03; addr < 0x78; addr++) {
        // Skip reserved addresses
        if (addr < 0x08 || (addr > 0x0F && addr < 0x1F) || addr > 0x77) {
            continue;
        }
        
        if (ioctl(i2c_fd, I2C_SLAVE, addr) < 0) {
            continue;
        }
        
        // Try to read from the device
        char buf;
        if (read(i2c_fd, &buf, 1) >= 0) {
            printf("Found I2C device at address 0x%02X\n", addr);
            found++;
        }
    }
    
    if (found == 0) {
        printf("No I2C devices found\n");
    } else {
        printf("Found %d I2C device(s)\n", found);
    }
}

// Select I2C channel on TCA9548A
int tca_select(int i2c_fd, uint8_t channel) {
    if (channel > 7) return -1;
    
    uint8_t data = 1 << channel;
    if (i2c_write_byte(i2c_fd, TCA9548A_ADDR, 0, data) < 0) {
        if (DEBUG) fprintf(stderr, "Failed to select channel %d on TCA9548A\n", channel);
        return -1;
    }
    
    return 0;
}

// Initialize MPU6050
int mpu6050_initialize(int i2c_fd) {
    // Wake up the MPU6050 (clear sleep bit)
    if (i2c_write_byte(i2c_fd, MPU6050_ADDR, MPU6050_REG_PWR_MGMT_1, 0) < 0) {
        if (DEBUG) fprintf(stderr, "Failed to initialize MPU6050\n");
        return -1;
    }
    
    return 0;
}

// Test MPU6050 connection
int mpu6050_test_connection(int i2c_fd) {
    uint8_t who_am_i;
    
    if (i2c_read_byte(i2c_fd, MPU6050_ADDR, MPU6050_REG_WHO_AM_I, &who_am_i) < 0) {
        if (DEBUG) fprintf(stderr, "Failed to read WHO_AM_I register\n");
        return 0;
    }
    
    if (DEBUG) printf("WHO_AM_I register value: 0x%02X\n", who_am_i);
    
    return (who_am_i == 0x68);  // MPU6050 should return 0x68
}

// Get accelerometer and gyroscope values
int mpu6050_get_motion_6(int i2c_fd, int16_t *ax, int16_t *ay, int16_t *az, int16_t *gx, int16_t *gy, int16_t *gz) {
    uint8_t buffer[14];
    
    if (i2c_read_bytes(i2c_fd, MPU6050_ADDR, MPU6050_REG_ACCEL_XOUT_H, buffer, 14) < 0) {
        if (DEBUG) fprintf(stderr, "Failed to read motion data\n");
        return -1;
    }
    
    *ax = (((int16_t)buffer[0]) << 8) | buffer[1];
    *ay = (((int16_t)buffer[2]) << 8) | buffer[3];
    *az = (((int16_t)buffer[4]) << 8) | buffer[5];
    *gx = (((int16_t)buffer[8]) << 8) | buffer[9];
    *gy = (((int16_t)buffer[10]) << 8) | buffer[11];
    *gz = (((int16_t)buffer[12]) << 8) | buffer[13];
    
    return 0;
}

// Write a byte to an I2C device
int i2c_write_byte(int i2c_fd, uint8_t dev_addr, uint8_t reg_addr, uint8_t data) {
    // Set device address
    if (ioctl(i2c_fd, I2C_SLAVE, dev_addr) < 0) {
        fprintf(stderr, "Failed to set I2C slave address 0x%02X: %s\n", dev_addr, strerror(errno));
        return -1;
    }
    
    // For direct register write, we use a simple buffer
    uint8_t buffer[2];
    buffer[0] = reg_addr;
    buffer[1] = data;
    
    if (write(i2c_fd, buffer, 2) != 2) {
        fprintf(stderr, "Failed to write to I2C device 0x%02X: %s\n", dev_addr, strerror(errno));
        return -1;
    }
    
    return 0;
}

// Read a byte from an I2C device
int i2c_read_byte(int i2c_fd, uint8_t dev_addr, uint8_t reg_addr, uint8_t *data) {
    // Set device address
    if (ioctl(i2c_fd, I2C_SLAVE, dev_addr) < 0) {
        fprintf(stderr, "Failed to set I2C slave address 0x%02X: %s\n", dev_addr, strerror(errno));
        return -1;
    }
    
    // Write the register address
    if (write(i2c_fd, &reg_addr, 1) != 1) {
        fprintf(stderr, "Failed to write register address to I2C device 0x%02X: %s\n", dev_addr, strerror(errno));
        return -1;
    }
    
    // Read the data
    if (read(i2c_fd, data, 1) != 1) {
        fprintf(stderr, "Failed to read from I2C device 0x%02X: %s\n", dev_addr, strerror(errno));
        return -1;
    }
    
    return 0;
}

// Read multiple bytes from an I2C device
int i2c_read_bytes(int i2c_fd, uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, size_t length) {
    // Set device address
    if (ioctl(i2c_fd, I2C_SLAVE, dev_addr) < 0) {
        fprintf(stderr, "Failed to set I2C slave address 0x%02X: %s\n", dev_addr, strerror(errno));
        return -1;
    }
    
    // Write the register address
    if (write(i2c_fd, &reg_addr, 1) != 1) {
        fprintf(stderr, "Failed to write register address to I2C device 0x%02X: %s\n", dev_addr, strerror(errno));
        return -1;
    }
    
    // Read the data
    if (read(i2c_fd, data, length) != (int)length) {
        fprintf(stderr, "Failed to read %zu bytes from I2C device 0x%02X: %s\n", length, dev_addr, strerror(errno));
        return -1;
    }
    
    return 0;
}
