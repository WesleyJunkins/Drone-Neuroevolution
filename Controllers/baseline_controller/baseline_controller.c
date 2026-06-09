//This is the final version of the baseline controller that works.
//Please use this controller for the baseline tests. April 6, 2026.

/*
 * Copyright 2022 Bitcraze AB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 *  ...........       ____  _ __
 *  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 *  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 *  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 *
 * @file crazyflie_controller.c
 * Description: Controls the crazyflie in webots
 * Author:      Kimberly McGuire (Bitcraze AB)
 */

 #include <math.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <unistd.h>
 #include <fcntl.h>
 #include <sys/socket.h>
 #include <netinet/in.h>
 #include <arpa/inet.h>
 #include <stdint.h>
 
 #include <webots/camera.h>
 #include <webots/distance_sensor.h>
 #include <webots/gps.h>
 #include <webots/gyro.h>
 #include <webots/inertial_unit.h>
 #include <webots/keyboard.h>
 #include <webots/motor.h>
 #include <webots/robot.h>
 #include <webots/supervisor.h>

// Add external controller
#include "pid_controller.h"

// ── ADJUSTABLE TEST PARAMETERS ───────────────────────────────────────────────
// SPEED_SCALAR is the startup default only — the receiver overwrites it at
// runtime via new_speed_scalar in the first JSON command after each reset.
// No recompile needed between tests; change only if you want a different default.
// YAW_SCALAR sets yaw rate (rad/s) — not sent by receiver, change here if needed.
#define SPEED_SCALAR  0.6f
#define YAW_SCALAR    1.55f
// ─────────────────────────────────────────────────────────────────────────────

#define FLYING_ALTITUDE 1.0
#define SOCKET_PORT 8080
#define BUFFER_SIZE 1024

// Starting position bounding box for lap detection
#define START_BOX_X_MIN -0.5
#define START_BOX_X_MAX 0.5
#define START_BOX_Y_MIN -0.5
#define START_BOX_Y_MAX 0.5

// Structure to hold control commands from JSON
typedef struct {
    int forward;
    int backward;
    int left;
    int right;
    int yaw_increase;
    int yaw_decrease;
    int height_diff_increase;
    int height_diff_decrease;
    int reset_simulation;
    float new_speed_scalar;  // 0.0 means no update
    float new_yaw_scalar;    // 0.0 means no update
} control_commands_json_t;

// Global socket for communication
static int g_socket = -1;

// Function to convert RGBA image to grayscale brightness values (0-255)
int get_brightness(const unsigned char *rgba_pixel) {
    // Convert RGBA to grayscale using standard luminance formula
    // R*0.299 + G*0.587 + B*0.114
    int r = rgba_pixel[0];
    int g = rgba_pixel[1];
    int b = rgba_pixel[2];
    return (int)(r * 0.299 + g * 0.587 + b * 0.114);
}

// Function to check if drone position is within the starting box
int is_in_starting_box(double x, double y) {
    return (x >= START_BOX_X_MIN && x <= START_BOX_X_MAX &&
            y >= START_BOX_Y_MIN && y <= START_BOX_Y_MAX);
}

// Function to send 2D array to port with lap count and speed scalar
void send_2d_array(WbDeviceTag camera, int lap_count, float speed_scalar) {
    // Get camera image
    const unsigned char *image = wb_camera_get_image(camera);
    if (!image) {
        // printf("Cannot send array: no image data available\n");
        return;
    }
    
    int width = wb_camera_get_width(camera);
    int height = wb_camera_get_height(camera);
    int channels = 4; // RGBA
    
    // printf("Converting image to 2D array: %dx%d\n", width, height);
    
    // Create 2D array
    uint8_t *array_2d = malloc(width * height * sizeof(uint8_t));
    if (!array_2d) {
        return;
    }

    // Convert RGBA image to grayscale (0-255) — no further quantization
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = (y * width + x) * channels;
            array_2d[y * width + x] = (uint8_t)get_brightness(&image[pixel_index]);
        }
    }
    
    // Print the 2D array before sending
    // printf("\n=== CONTROLLER SENDING 2D ARRAY (%dx%d) ===\n", width, height);
    // printf("0=black, 1=dark_gray, 2=medium_gray, 3=light_gray, 4=white\n");
    // printf("-");
    // for (int i = 0; i < width * 2 + 8; i++) printf("-");
    // printf("\n");
    
    // for (int y = 0; y < height; y++) {
    //     for (int x = 0; x < width; x++) {
    //         printf("%d ", array_2d[y * width + x]);
    //     }
    //     printf("\n");
    // }
    
    // printf("-");
    // for (int i = 0; i < width * 2 + 8; i++) printf("-");
    // printf("\n");
    
    // Send array data to port
    int array_size = width * height * sizeof(uint8_t);
    send(g_socket, (const char*)array_2d, array_size, 0);

    // Send lap count as a 32-bit integer (4 bytes)
    int32_t lap_count_int = (int32_t)lap_count;
    send(g_socket, (const char*)&lap_count_int, sizeof(int32_t), 0);

    // Send speed scalar as a 32-bit float (4 bytes)
    send(g_socket, (const char*)&speed_scalar, sizeof(float), 0);

    // Clean up
    free(array_2d);
}

// Function to parse JSON control commands
control_commands_json_t parse_control_json(const char* json_str) {
    control_commands_json_t commands = {0};
    
    // More robust JSON parsing - handle spaces and different formats
    if (strstr(json_str, "\"forward\":1") || strstr(json_str, "\"forward\": 1")) commands.forward = 1;
    if (strstr(json_str, "\"backward\":1") || strstr(json_str, "\"backward\": 1")) commands.backward = 1;
    if (strstr(json_str, "\"left\":1") || strstr(json_str, "\"left\": 1")) commands.left = 1;
    if (strstr(json_str, "\"right\":1") || strstr(json_str, "\"right\": 1")) commands.right = 1;
    if (strstr(json_str, "\"yaw_increase\":1") || strstr(json_str, "\"yaw_increase\": 1")) commands.yaw_increase = 1;
    if (strstr(json_str, "\"yaw_decrease\":1") || strstr(json_str, "\"yaw_decrease\": 1")) commands.yaw_decrease = 1;
    if (strstr(json_str, "\"height_diff_increase\":1") || strstr(json_str, "\"height_diff_increase\": 1")) commands.height_diff_increase = 1;
    if (strstr(json_str, "\"height_diff_decrease\":1") || strstr(json_str, "\"height_diff_decrease\": 1")) commands.height_diff_decrease = 1;
    if (strstr(json_str, "\"reset_simulation\":1") || strstr(json_str, "\"reset_simulation\": 1")) commands.reset_simulation = 1;

    const char *scalar_ptr = strstr(json_str, "\"new_speed_scalar\":");
    if (scalar_ptr) {
        scalar_ptr += strlen("\"new_speed_scalar\":");
        while (*scalar_ptr == ' ') scalar_ptr++;
        commands.new_speed_scalar = (float)atof(scalar_ptr);
    }

    const char *yaw_scalar_ptr = strstr(json_str, "\"new_yaw_scalar\":");
    if (yaw_scalar_ptr) {
        yaw_scalar_ptr += strlen("\"new_yaw_scalar\":");
        while (*yaw_scalar_ptr == ' ') yaw_scalar_ptr++;
        commands.new_yaw_scalar = (float)atof(yaw_scalar_ptr);
    }

    return commands;
}

// Function to check for incoming commands
control_commands_json_t check_for_commands() {
    control_commands_json_t commands = {0};
    char buffer[BUFFER_SIZE];
    
    if (g_socket >= 0) {
        int bytes_received = recv(g_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            commands = parse_control_json(buffer);
        }
    }
    
    return commands;
}

// Function to initialize socket connection
int init_socket_connection() {
    g_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (g_socket == -1) {
        // printf("Failed to create socket\n");
        return -1;
    }
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(SOCKET_PORT);
    
    if (connect(g_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        // printf("Failed to connect to port %d\n", SOCKET_PORT);
        close(g_socket);
        return -1;
    }
    
    // Set socket to non-blocking
    int flags = fcntl(g_socket, F_GETFL, 0);
    fcntl(g_socket, F_SETFL, flags | O_NONBLOCK);
    
    // printf("Connected to port %d\n", SOCKET_PORT);
    return g_socket;
}


int main(int argc, char **argv) {
  wb_robot_init();

  const int timestep = (int)wb_robot_get_basic_time_step();

  // Initialize socket connection
  init_socket_connection();

  // Initialize motors
  WbDeviceTag m1_motor = wb_robot_get_device("m1_motor");
  wb_motor_set_position(m1_motor, INFINITY);
  wb_motor_set_velocity(m1_motor, -1.0);
  WbDeviceTag m2_motor = wb_robot_get_device("m2_motor");
  wb_motor_set_position(m2_motor, INFINITY);
  wb_motor_set_velocity(m2_motor, 1.0);
  WbDeviceTag m3_motor = wb_robot_get_device("m3_motor");
  wb_motor_set_position(m3_motor, INFINITY);
  wb_motor_set_velocity(m3_motor, -1.0);
  WbDeviceTag m4_motor = wb_robot_get_device("m4_motor");
  wb_motor_set_position(m4_motor, INFINITY);
  wb_motor_set_velocity(m4_motor, 1.0);

  // Initialize sensors
  WbDeviceTag imu = wb_robot_get_device("inertial_unit");
  wb_inertial_unit_enable(imu, timestep);
  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, timestep);
  wb_keyboard_enable(timestep);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, timestep);
  WbDeviceTag camera = wb_robot_get_device("down_camera");
  wb_camera_enable(camera, timestep);
  WbDeviceTag range_front = wb_robot_get_device("range_front");
  wb_distance_sensor_enable(range_front, timestep);
  WbDeviceTag range_left = wb_robot_get_device("range_left");
  wb_distance_sensor_enable(range_left, timestep);
  WbDeviceTag range_back = wb_robot_get_device("range_back");
  wb_distance_sensor_enable(range_back, timestep);
  WbDeviceTag range_right = wb_robot_get_device("range_right");
  wb_distance_sensor_enable(range_right, timestep);

  // Wait for 2 seconds
  while (wb_robot_step(timestep) != -1) {
    if (wb_robot_get_time() > 2.0)
      break;
  }

  // Initialize variables
  actual_state_t actual_state = {0};
  desired_state_t desired_state = {0};
  double past_x_global = 0;
  double past_y_global = 0;
  double past_time = wb_robot_get_time();

  // Initialize PID gains.
  gains_pid_t gains_pid;
  gains_pid.kp_att_y = 1;
  gains_pid.kd_att_y = 0.5;
  gains_pid.kp_att_rp = 0.5;
  gains_pid.kd_att_rp = 0.1;
  gains_pid.kp_vel_xy = 2;
  gains_pid.kd_vel_xy = 0.5;
  gains_pid.kp_z = 10;
  gains_pid.ki_z = 5;
  gains_pid.kd_z = 5;
  init_pid_attitude_fixed_height_controller();

  double height_desired = FLYING_ALTITUDE;

  // Initialize struct for motor power
  motor_power_t motor_power;

  // Initialize lap detection variables
  static int was_in_box = 1;  // Start as 1: drone spawns inside the box
  static int lap_count = 0;  // Track total number of completed laps

  // Mutable scalars — updated at runtime from receiver JSON fields.
  // Receiver sends correct values on first command after each reset.
  float speed_scalar = SPEED_SCALAR;
  float yaw_scalar   = YAW_SCALAR;

  // printf("\n");

  // printf("====== Active =======\n");

  while (wb_robot_step(timestep) != -1) {
    const double dt = wb_robot_get_time() - past_time;

    // Get measurements
    actual_state.roll = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
    actual_state.pitch = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
    actual_state.yaw_rate = wb_gyro_get_values(gyro)[2];
    actual_state.altitude = wb_gps_get_values(gps)[2];
    double x_global = wb_gps_get_values(gps)[0];
    double vx_global = (x_global - past_x_global) / dt;
    double y_global = wb_gps_get_values(gps)[1];
    double vy_global = (y_global - past_y_global) / dt;

    // Get body fixed velocities
    double actualYaw = wb_inertial_unit_get_roll_pitch_yaw(imu)[2];
    double cosyaw = cos(actualYaw);
    double sinyaw = sin(actualYaw);
    actual_state.vx = vx_global * cosyaw + vy_global * sinyaw;
    actual_state.vy = -vx_global * sinyaw + vy_global * cosyaw;

    // Initialize values
    desired_state.roll = 0;
    desired_state.pitch = 0;
    desired_state.vx = 0;
    desired_state.vy = 0;
    desired_state.yaw_rate = 0;
    desired_state.altitude = 1.0;

    double forward_desired = 0;
    double sideways_desired = 0;
    double yaw_desired = 0;
    double height_diff_desired = 0;

    // Lap detection: Check if drone is in starting box
    int is_in_box = is_in_starting_box(x_global, y_global);
    
    // Detect lap completion: entering box after being outside
    if (!was_in_box && is_in_box) {
        lap_count++;  // Increment lap counter
        printf("LAP COMPLETED! Total laps: %d, Drone entered starting box at (%.2f, %.2f)\n", lap_count, x_global, y_global);
    }
    
    // DEBUG
    // printf("Is in box: %d, was in box: %d, x_global: %f, y_global: %f\n", is_in_box, was_in_box, x_global, y_global);
    // printf("Lap count: %d\n", lap_count);
    
    // Update state for next timestep
    was_in_box = is_in_box;

    // Send ready-up 2D array 5 seconds after startup (allow drone to get in air)
    static int startup_counter = 0;
    startup_counter++;
    if (startup_counter == 500) {  // fires once after 500 * basicTimeStep ms of simulation time
        send_2d_array(camera, lap_count, speed_scalar);
    }

    // Static array to hold the last received commands (persists between iterations)
    static int active_commands[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    static int command_received = 0;  // Flag to track if a command was received and needs processing
    
    // Check for incoming commands from the socket
    control_commands_json_t socket_commands = check_for_commands();
    
    // Update runtime scalars if the receiver sent new values
    if (socket_commands.new_speed_scalar > 0.0f) {
        speed_scalar = socket_commands.new_speed_scalar;
    }
    if (socket_commands.new_yaw_scalar > 0.0f) {
        yaw_scalar = socket_commands.new_yaw_scalar;
    }

    // Handle simulation reset command
    if (socket_commands.reset_simulation) {
        // Reset simulation using supervisor
        // printf("Reset simulation requested - restarting simulation\n");
        
        // Get reference to the current robot node
        WbNodeRef robot_node = wb_supervisor_node_get_self();
        
        // Restart the simulation
        wb_supervisor_simulation_reset();
        
        // Restart this controller
        wb_supervisor_node_restart_controller(robot_node);
        
        return 0;  // Exit the current instance
    }
    
    // Update active_commands array only if we received new commands
    if (socket_commands.forward || socket_commands.backward || socket_commands.left || 
        socket_commands.right || socket_commands.yaw_increase || socket_commands.yaw_decrease ||
        socket_commands.height_diff_increase || socket_commands.height_diff_decrease) {
        
        // printf("Received new commands, updating active_commands array...\n");
        active_commands[0] = socket_commands.forward;
        active_commands[1] = socket_commands.backward;
        active_commands[2] = socket_commands.left;
        active_commands[3] = socket_commands.right;
        active_commands[4] = socket_commands.yaw_increase;
        active_commands[5] = socket_commands.yaw_decrease;
        active_commands[6] = socket_commands.height_diff_increase;
        active_commands[7] = socket_commands.height_diff_decrease;
        active_commands[8] = 0; // unused
        
        command_received = 1;  // Set flag to indicate a command was received
        
        // printf("Active commands updated: %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
        //        active_commands[0], active_commands[1], active_commands[2], active_commands[3],
        //        active_commands[4], active_commands[5], active_commands[6], active_commands[7], active_commands[8]);
    }
    
    // Reset all command values
    forward_desired = 0;
    sideways_desired = 0;
    yaw_desired = 0;
    height_diff_desired = 0;

    // printf("Active commands: %d, %d, %d, %d, %d, %d, %d, %d, %d\n",
    //        active_commands[0], active_commands[1], active_commands[2], active_commands[3],
    //        active_commands[4], active_commands[5], active_commands[6], active_commands[7], active_commands[8]);
    
    // Apply all active commands concurrently
    for (int i = 0; i < 9; i++) {
      if (active_commands[i] == 1) {  // Only process active commands
        switch (i + 1) {  // Convert array index to command number (0->1, 1->2, etc.)
          case 1:
            forward_desired += 1.0 * speed_scalar;  // Use += to combine multiple commands
            break;
          case 2:
            forward_desired -= 1.0 * speed_scalar;
            break;
          case 3:
            sideways_desired -= 1.0 * speed_scalar;
            break;
          case 4:
            sideways_desired += 1.0 * speed_scalar;
            break;
          case 5:
            yaw_desired += 1.0 * yaw_scalar;
            break;
          case 6:
            yaw_desired -= 1.0 * yaw_scalar;
            break;
          case 7:
            height_diff_desired += 0.1;
            break;
          case 8:
            height_diff_desired -= 0.1;
            break;
        }
      }
    }

    height_desired += height_diff_desired * dt;

    // Send new 2D array only if a command was received and processed
    if (command_received) {
        send_2d_array(camera, lap_count, speed_scalar);
        command_received = 0;  // Reset flag after sending array
    }

    // Example how to get sensor data
    // range_front_value = wb_distance_sensor_get_value(range_front));
    // const unsigned char *image = wb_camera_get_image(camera);

    desired_state.yaw_rate = yaw_desired;

    // PID velocity controller with fixed height
    desired_state.vy = sideways_desired;
    desired_state.vx = forward_desired;
    desired_state.altitude = height_desired;
    pid_velocity_fixed_height_controller(actual_state, &desired_state, gains_pid, dt, &motor_power);

    // Setting motorspeed
    wb_motor_set_velocity(m1_motor, -motor_power.m1);
    wb_motor_set_velocity(m2_motor, motor_power.m2);
    wb_motor_set_velocity(m3_motor, -motor_power.m3);
    wb_motor_set_velocity(m4_motor, motor_power.m4);

    // Save past time for next time step
    past_time = wb_robot_get_time();
    past_x_global = x_global;
    past_y_global = y_global;
  };

  // Clean up socket
  if (g_socket >= 0) {
    close(g_socket);
  }

  wb_robot_cleanup();

  return 0;
}