// NE Controller for Phase 4 testing.
// Based on baseline_controller.c — same sensors, PID, lap detection, and startup sequence.
// Supports three command modes selected at RUNTIME via the "command_mode" JSON field:
//   0 = binary commands   [forward, yaw_increase, yaw_decrease]  (same as baseline)
//   1 = continuous values [forward_desired, yaw_desired]         (set PID targets directly)
//   2 = motor velocities  [m1, m2, m3, m4]                      (bypass PID entirely)
// The receiver sends "command_mode" in every JSON packet — no recompile needed between models.
// By Wesley Junkins

/*
 * Copyright 2022 Bitcraze AB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
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

#include "pid_controller.h"

// ── ADJUSTABLE PARAMETERS ────────────────────────────────────────────────────
// SPEED_SCALAR / YAW_SCALAR are startup defaults only.
// The receiver overwrites them via new_speed_scalar / new_yaw_scalar in the
// first JSON command after each simulation reset — no recompile needed.
#define SPEED_SCALAR  0.6f
#define YAW_SCALAR    1.55f

// Set TEST_MODE to skip the full 500-step hover stabilization (16s per test).
// In TEST_MODE the drone sends its first image after 50 steps (~1.6s) instead.
// Enable for quick pipeline verification; disable for real test runs.
// #define TEST_MODE

// ─────────────────────────────────────────────────────────────────────────────

#define FLYING_ALTITUDE 1.0
#define SOCKET_PORT 8080
#define BUFFER_SIZE 1024

#define START_BOX_X_MIN -0.5
#define START_BOX_X_MAX  0.5
#define START_BOX_Y_MIN -0.5
#define START_BOX_Y_MAX  0.5

// Extended command struct — fields for all three command modes.
typedef struct {
    // Runtime mode sent by receiver every step (0=binary, 1=continuous, 2=motor)
    int command_mode;
    // Binary mode (0)
    int forward;
    int backward;
    int left;
    int right;
    int yaw_increase;
    int yaw_decrease;
    int height_diff_increase;
    int height_diff_decrease;
    // Continuous mode (COMMAND_MODE 1)
    float forward_desired;
    float yaw_desired;
    // Motor mode (COMMAND_MODE 2)
    float m1;
    float m2;
    float m3;
    float m4;
    // Common
    int reset_simulation;
    float new_speed_scalar;
    float new_yaw_scalar;
} ne_commands_t;

static int g_socket = -1;

int get_brightness(const unsigned char *rgba_pixel) {
    int r = rgba_pixel[0];
    int g = rgba_pixel[1];
    int b = rgba_pixel[2];
    return (int)(r * 0.299 + g * 0.587 + b * 0.114);
}

int is_in_starting_box(double x, double y) {
    return (x >= START_BOX_X_MIN && x <= START_BOX_X_MAX &&
            y >= START_BOX_Y_MIN && y <= START_BOX_Y_MAX);
}

void send_2d_array(WbDeviceTag camera, int lap_count, float speed_scalar) {
    const unsigned char *image = wb_camera_get_image(camera);
    if (!image) return;

    int width  = wb_camera_get_width(camera);
    int height = wb_camera_get_height(camera);

    uint8_t *array_2d = malloc(width * height * sizeof(uint8_t));
    if (!array_2d) return;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = (y * width + x) * 4;  // RGBA
            array_2d[y * width + x] = (uint8_t)get_brightness(&image[pixel_index]);
        }
    }

    int array_size = width * height * sizeof(uint8_t);
    send(g_socket, (const char*)array_2d, array_size, 0);

    int32_t lap_count_int = (int32_t)lap_count;
    send(g_socket, (const char*)&lap_count_int, sizeof(int32_t), 0);
    send(g_socket, (const char*)&speed_scalar, sizeof(float), 0);

    free(array_2d);
}

ne_commands_t parse_ne_json(const char *json_str) {
    ne_commands_t cmd = {0};

    // Binary flags — same robust pattern as baseline
    if (strstr(json_str, "\"forward\":1")              || strstr(json_str, "\"forward\": 1"))              cmd.forward = 1;
    if (strstr(json_str, "\"backward\":1")             || strstr(json_str, "\"backward\": 1"))             cmd.backward = 1;
    if (strstr(json_str, "\"left\":1")                 || strstr(json_str, "\"left\": 1"))                 cmd.left = 1;
    if (strstr(json_str, "\"right\":1")                || strstr(json_str, "\"right\": 1"))                cmd.right = 1;
    if (strstr(json_str, "\"yaw_increase\":1")         || strstr(json_str, "\"yaw_increase\": 1"))         cmd.yaw_increase = 1;
    if (strstr(json_str, "\"yaw_decrease\":1")         || strstr(json_str, "\"yaw_decrease\": 1"))         cmd.yaw_decrease = 1;
    if (strstr(json_str, "\"height_diff_increase\":1") || strstr(json_str, "\"height_diff_increase\": 1")) cmd.height_diff_increase = 1;
    if (strstr(json_str, "\"height_diff_decrease\":1") || strstr(json_str, "\"height_diff_decrease\": 1")) cmd.height_diff_decrease = 1;
    if (strstr(json_str, "\"reset_simulation\":1")     || strstr(json_str, "\"reset_simulation\": 1"))     cmd.reset_simulation = 1;

    // Continuous mode fields
    const char *fwd_ptr = strstr(json_str, "\"forward_desired\":");
    if (fwd_ptr) {
        fwd_ptr += strlen("\"forward_desired\":");
        while (*fwd_ptr == ' ') fwd_ptr++;
        cmd.forward_desired = (float)atof(fwd_ptr);
    }
    const char *yaw_ptr = strstr(json_str, "\"yaw_desired\":");
    if (yaw_ptr) {
        yaw_ptr += strlen("\"yaw_desired\":");
        while (*yaw_ptr == ' ') yaw_ptr++;
        cmd.yaw_desired = (float)atof(yaw_ptr);
    }

    // Motor mode fields
    const char *m1_ptr = strstr(json_str, "\"m1\":");
    if (m1_ptr) { m1_ptr += strlen("\"m1\":"); while (*m1_ptr == ' ') m1_ptr++; cmd.m1 = (float)atof(m1_ptr); }
    const char *m2_ptr = strstr(json_str, "\"m2\":");
    if (m2_ptr) { m2_ptr += strlen("\"m2\":"); while (*m2_ptr == ' ') m2_ptr++; cmd.m2 = (float)atof(m2_ptr); }
    const char *m3_ptr = strstr(json_str, "\"m3\":");
    if (m3_ptr) { m3_ptr += strlen("\"m3\":"); while (*m3_ptr == ' ') m3_ptr++; cmd.m3 = (float)atof(m3_ptr); }
    const char *m4_ptr = strstr(json_str, "\"m4\":");
    if (m4_ptr) { m4_ptr += strlen("\"m4\":"); while (*m4_ptr == ' ') m4_ptr++; cmd.m4 = (float)atof(m4_ptr); }

    // Command mode (sent by receiver every step)
    const char *mode_ptr = strstr(json_str, "\"command_mode\":");
    if (mode_ptr) {
        mode_ptr += strlen("\"command_mode\":");
        while (*mode_ptr == ' ') mode_ptr++;
        cmd.command_mode = (int)atoi(mode_ptr);
    }

    // Scalar updates
    const char *spd_ptr = strstr(json_str, "\"new_speed_scalar\":");
    if (spd_ptr) {
        spd_ptr += strlen("\"new_speed_scalar\":");
        while (*spd_ptr == ' ') spd_ptr++;
        cmd.new_speed_scalar = (float)atof(spd_ptr);
    }
    const char *ys_ptr = strstr(json_str, "\"new_yaw_scalar\":");
    if (ys_ptr) {
        ys_ptr += strlen("\"new_yaw_scalar\":");
        while (*ys_ptr == ' ') ys_ptr++;
        cmd.new_yaw_scalar = (float)atof(ys_ptr);
    }

    return cmd;
}

ne_commands_t check_for_commands() {
    ne_commands_t cmd = {0};
    cmd.command_mode = -1;  // sentinel: -1 means no packet received this frame
    char buffer[BUFFER_SIZE];
    if (g_socket >= 0) {
        int bytes_received = recv(g_socket, buffer, BUFFER_SIZE - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            cmd = parse_ne_json(buffer);
        }
    }
    return cmd;
}

int init_socket_connection() {
    g_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (g_socket == -1) return -1;

    struct sockaddr_in server_addr;
    server_addr.sin_family      = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port        = htons(SOCKET_PORT);

    if (connect(g_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(g_socket);
        return -1;
    }

    int flags = fcntl(g_socket, F_GETFL, 0);
    fcntl(g_socket, F_SETFL, flags | O_NONBLOCK);
    return g_socket;
}


int main(int argc, char **argv) {
    wb_robot_init();

    const int timestep = (int)wb_robot_get_basic_time_step();

    init_socket_connection();

    // Motors
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

    // Sensors
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

    // 2-second stabilization wait (identical to baseline)
    while (wb_robot_step(timestep) != -1) {
        if (wb_robot_get_time() > 2.0) break;
    }

    actual_state_t  actual_state  = {0};
    desired_state_t desired_state = {0};
    double past_x_global = 0;
    double past_y_global = 0;
    double past_time     = wb_robot_get_time();

    gains_pid_t gains_pid;
    gains_pid.kp_att_y  = 1;
    gains_pid.kd_att_y  = 0.5;
    gains_pid.kp_att_rp = 0.5;
    gains_pid.kd_att_rp = 0.1;
    gains_pid.kp_vel_xy = 2;
    gains_pid.kd_vel_xy = 0.5;
    gains_pid.kp_z      = 10;
    gains_pid.ki_z      = 5;
    gains_pid.kd_z      = 5;
    init_pid_attitude_fixed_height_controller();

    double height_desired = FLYING_ALTITUDE;
    motor_power_t motor_power;

    static int was_in_box = 1;
    static int lap_count  = 0;

    float speed_scalar = SPEED_SCALAR;
    float yaw_scalar   = YAW_SCALAR;

    // Persistent state for all three modes — runtime branching decides which to use.
    // current_mode is updated from the receiver's command_mode field each step.
    int   current_mode    = 0;  // 0=binary, 1=continuous, 2=motor
    float last_fwd_desired = 0.0f;
    float last_yaw_desired = 0.0f;
    float last_m1 = 0.0f, last_m2 = 0.0f, last_m3 = 0.0f, last_m4 = 0.0f;

    while (wb_robot_step(timestep) != -1) {
        const double dt = wb_robot_get_time() - past_time;

        // Sensor readings
        actual_state.roll     = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
        actual_state.pitch    = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
        actual_state.yaw_rate = wb_gyro_get_values(gyro)[2];
        actual_state.altitude = wb_gps_get_values(gps)[2];
        double x_global  = wb_gps_get_values(gps)[0];
        double vx_global = (x_global - past_x_global) / dt;
        double y_global  = wb_gps_get_values(gps)[1];
        double vy_global = (y_global - past_y_global) / dt;

        double actualYaw = wb_inertial_unit_get_roll_pitch_yaw(imu)[2];
        actual_state.vx = vx_global * cos(actualYaw) + vy_global * sin(actualYaw);
        actual_state.vy = -vx_global * sin(actualYaw) + vy_global * cos(actualYaw);

        desired_state.roll      = 0;
        desired_state.pitch     = 0;
        desired_state.vx        = 0;
        desired_state.vy        = 0;
        desired_state.yaw_rate  = 0;
        desired_state.altitude  = 1.0;

        // Lap detection (identical to baseline)
        int is_in_box = is_in_starting_box(x_global, y_global);
        if (!was_in_box && is_in_box) {
            lap_count++;
            printf("LAP COMPLETED! Total laps: %d, position (%.2f, %.2f)\n", lap_count, x_global, y_global);
        }
        was_in_box = is_in_box;

        // First image after startup stabilization (fires once).
        // Normal: 500 steps (~16s).  TEST_MODE: 50 steps (~1.6s).
#ifdef TEST_MODE
        static int startup_counter = 0;
        startup_counter++;
        if (startup_counter == 50) {
            send_2d_array(camera, lap_count, speed_scalar);
        }
#else
        static int startup_counter = 0;
        startup_counter++;
        if (startup_counter == 500) {
            send_2d_array(camera, lap_count, speed_scalar);
        }
#endif

        static int command_received = 0;
        static int active_commands[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        ne_commands_t socket_commands = check_for_commands();

        // Update runtime mode whenever a packet was actually received.
        // command_mode == -1 means no packet this frame (EAGAIN on non-blocking socket).
        int got_packet = (socket_commands.command_mode >= 0);
        if (got_packet)
            current_mode = socket_commands.command_mode;

        // Runtime scalar updates (all modes)
        if (socket_commands.new_speed_scalar > 0.0f) speed_scalar = socket_commands.new_speed_scalar;
        if (socket_commands.new_yaw_scalar   > 0.0f) yaw_scalar   = socket_commands.new_yaw_scalar;

        // Simulation reset (all modes)
        if (socket_commands.reset_simulation) {
            WbNodeRef robot_node = wb_supervisor_node_get_self();
            wb_supervisor_simulation_reset();
            wb_supervisor_node_restart_controller(robot_node);
            return 0;
        }

        // ── MODE 0: Binary commands (ycommand models) ──────────────────────────
        if (current_mode == 0) {
            if (got_packet) {
                // Always update latched commands — including all-zero (hover/do-nothing)
                active_commands[0] = socket_commands.forward;
                active_commands[1] = socket_commands.backward;
                active_commands[2] = socket_commands.left;
                active_commands[3] = socket_commands.right;
                active_commands[4] = socket_commands.yaw_increase;
                active_commands[5] = socket_commands.yaw_decrease;
                active_commands[6] = socket_commands.height_diff_increase;
                active_commands[7] = socket_commands.height_diff_decrease;
                active_commands[8] = 0;
                command_received = 1;
            }

            double forward_desired    = 0;
            double sideways_desired   = 0;
            double yaw_desired_val    = 0;
            double height_diff_desired = 0;

            for (int i = 0; i < 9; i++) {
                if (active_commands[i] == 1) {
                    switch (i + 1) {
                        case 1: forward_desired    += 1.0 * speed_scalar; break;
                        case 2: forward_desired    -= 1.0 * speed_scalar; break;
                        case 3: sideways_desired   -= 1.0 * speed_scalar; break;
                        case 4: sideways_desired   += 1.0 * speed_scalar; break;
                        case 5: yaw_desired_val    += 1.0 * yaw_scalar;   break;
                        case 6: yaw_desired_val    -= 1.0 * yaw_scalar;   break;
                        case 7: height_diff_desired += 0.1; break;
                        case 8: height_diff_desired -= 0.1; break;
                    }
                }
            }

            height_desired += height_diff_desired * dt;

            if (command_received) {
                send_2d_array(camera, lap_count, speed_scalar);
                command_received = 0;
            }

            desired_state.yaw_rate = yaw_desired_val;
            desired_state.vy       = sideways_desired;
            desired_state.vx       = forward_desired;
            desired_state.altitude = height_desired;
            pid_velocity_fixed_height_controller(actual_state, &desired_state, gains_pid, dt, &motor_power);

            wb_motor_set_velocity(m1_motor, -motor_power.m1);
            wb_motor_set_velocity(m2_motor,  motor_power.m2);
            wb_motor_set_velocity(m3_motor, -motor_power.m3);
            wb_motor_set_velocity(m4_motor,  motor_power.m4);

        // ── MODE 1: Continuous values (ycontinuous models) ─────────────────────
        } else if (current_mode == 1) {
            if (got_packet) {
                // Always update — model may validly output (0, 0) meaning hover
                last_fwd_desired = socket_commands.forward_desired;
                last_yaw_desired = socket_commands.yaw_desired;
                command_received = 1;
            }

            if (command_received) {
                send_2d_array(camera, lap_count, speed_scalar);
                command_received = 0;
            }

            desired_state.yaw_rate = (double)last_yaw_desired;
            desired_state.vy       = 0.0;
            desired_state.vx       = (double)last_fwd_desired;
            desired_state.altitude = height_desired;
            pid_velocity_fixed_height_controller(actual_state, &desired_state, gains_pid, dt, &motor_power);

            wb_motor_set_velocity(m1_motor, -motor_power.m1);
            wb_motor_set_velocity(m2_motor,  motor_power.m2);
            wb_motor_set_velocity(m3_motor, -motor_power.m3);
            wb_motor_set_velocity(m4_motor,  motor_power.m4);

        // ── MODE 2: Motor velocities (ymotors models — bypasses PID) ───────────
        } else if (current_mode == 2) {
            if (got_packet) {
                // Always update — model may validly output all-zero velocities
                last_m1 = socket_commands.m1;
                last_m2 = socket_commands.m2;
                last_m3 = socket_commands.m3;
                last_m4 = socket_commands.m4;
                command_received = 1;
            }

            if (command_received) {
                send_2d_array(camera, lap_count, speed_scalar);
                command_received = 0;
            }

            // Model is responsible for all 4 DOF including altitude.
            wb_motor_set_velocity(m1_motor, (double)last_m1);
            wb_motor_set_velocity(m2_motor, (double)last_m2);
            wb_motor_set_velocity(m3_motor, (double)last_m3);
            wb_motor_set_velocity(m4_motor, (double)last_m4);
        }

        past_time     = wb_robot_get_time();
        past_x_global = x_global;
        past_y_global = y_global;
    }

    if (g_socket >= 0) close(g_socket);
    wb_robot_cleanup();
    return 0;
}
