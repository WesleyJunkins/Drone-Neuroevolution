// Manual flight controller for wesDroneRL2.
// Pilot controls the drone via keyboard; all sensor and command data is
// sent to manual_receiver.py every timestep for training data collection.
// No commands are received back from the receiver.

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

// ── SET BEFORE EACH FLYING SESSION ───────────────────────────────────────────
// Change SPEED_SCALAR to the speed you want to fly at (0.1 – 1.0).
// Webots recompiles automatically when you reload the world.
// You can also tap + / - during flight to adjust by 0.1 steps.
#define SPEED_SCALAR 0.2f
// ─────────────────────────────────────────────────────────────────────────────

#define FLYING_ALTITUDE  1.0f
#define SOCKET_PORT      8080
#define PACKET_SIZE      4136   // 4096 image + 40 metadata bytes

#define START_BOX_X_MIN -0.5
#define START_BOX_X_MAX  0.5
#define START_BOX_Y_MIN -0.5
#define START_BOX_Y_MAX  0.5

static int g_socket = -1;
static int g_step   = 0;

static int get_brightness(const unsigned char *p) {
    return (int)(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114);
}

static int is_in_box(double x, double y) {
    return x >= START_BOX_X_MIN && x <= START_BOX_X_MAX &&
           y >= START_BOX_Y_MIN && y <= START_BOX_Y_MAX;
}

// Build and send the 4136-byte training data packet.
//
// Packet layout (little-endian):
//   [4096 bytes] uint8  grayscale image (64×64 flat, row-major)
//   [   4 bytes] int32  timestep counter
//   [   4 bytes] float  sim timestamp (seconds)
//   [   1 byte]  uint8  forward command (0/1)
//   [   1 byte]  uint8  yaw_increase command (0/1)
//   [   1 byte]  uint8  yaw_decrease command (0/1)
//   [   1 byte]  uint8  reset command (0/1)
//   [   4 bytes] float  forward_desired (m/s target)
//   [   4 bytes] float  yaw_desired (rad/s target)
//   [   4 bytes] float  m1 velocity (set on motor)
//   [   4 bytes] float  m2 velocity
//   [   4 bytes] float  m3 velocity
//   [   4 bytes] float  m4 velocity
//   [   4 bytes] float  speed_scalar
//   ─────────────────────────────────
//   4136 bytes total
static void send_packet(WbDeviceTag camera, float simtime,
                        uint8_t fwd, uint8_t yaw_inc, uint8_t yaw_dec, uint8_t rst,
                        float fwd_des, float yaw_des,
                        float m1, float m2, float m3, float m4,
                        float speed) {
    if (g_socket < 0) return;
    const unsigned char *img = wb_camera_get_image(camera);
    if (!img) return;

    int w = wb_camera_get_width(camera);
    int h = wb_camera_get_height(camera);

    uint8_t pkt[PACKET_SIZE];
    int o = 0;

    for (int py = 0; py < h; py++)
        for (int px = 0; px < w; px++)
            pkt[o++] = (uint8_t)get_brightness(&img[(py * w + px) * 4]);

    int32_t ts = (int32_t)g_step;
    memcpy(pkt + o, &ts,      4); o += 4;
    memcpy(pkt + o, &simtime, 4); o += 4;

    pkt[o++] = fwd;
    pkt[o++] = yaw_inc;
    pkt[o++] = yaw_dec;
    pkt[o++] = rst;

    memcpy(pkt + o, &fwd_des, 4); o += 4;
    memcpy(pkt + o, &yaw_des, 4); o += 4;
    memcpy(pkt + o, &m1,      4); o += 4;
    memcpy(pkt + o, &m2,      4); o += 4;
    memcpy(pkt + o, &m3,      4); o += 4;
    memcpy(pkt + o, &m4,      4); o += 4;
    memcpy(pkt + o, &speed,   4); o += 4;

    send(g_socket, pkt, PACKET_SIZE, 0);
}

static int init_socket(void) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port        = htons(SOCKET_PORT);

    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(s);
        printf("WARNING: Could not connect to receiver on port %d. "
               "Flight will proceed but data will NOT be recorded.\n", SOCKET_PORT);
        return -1;
    }

    // Non-blocking so send() never stalls the simulation step
    int flags = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, flags | O_NONBLOCK);
    printf("Connected to receiver on port %d.\n", SOCKET_PORT);
    return s;
}

int main(int argc, char **argv) {
    wb_robot_init();
    const int timestep = (int)wb_robot_get_basic_time_step();

    g_socket = init_socket();

    // Motors
    WbDeviceTag m1_motor = wb_robot_get_device("m1_motor");
    wb_motor_set_position(m1_motor, INFINITY); wb_motor_set_velocity(m1_motor, -1.0);
    WbDeviceTag m2_motor = wb_robot_get_device("m2_motor");
    wb_motor_set_position(m2_motor, INFINITY); wb_motor_set_velocity(m2_motor,  1.0);
    WbDeviceTag m3_motor = wb_robot_get_device("m3_motor");
    wb_motor_set_position(m3_motor, INFINITY); wb_motor_set_velocity(m3_motor, -1.0);
    WbDeviceTag m4_motor = wb_robot_get_device("m4_motor");
    wb_motor_set_position(m4_motor, INFINITY); wb_motor_set_velocity(m4_motor,  1.0);

    // Sensors
    WbDeviceTag imu = wb_robot_get_device("inertial_unit");
    wb_inertial_unit_enable(imu, timestep);
    WbDeviceTag gps = wb_robot_get_device("gps");
    wb_gps_enable(gps, timestep);
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
    wb_keyboard_enable(timestep);

    // Stabilization wait — let drone reach hover altitude before recording
    while (wb_robot_step(timestep) != -1)
        if (wb_robot_get_time() > 2.0) break;

    // PID init
    actual_state_t  actual_state  = {0};
    desired_state_t desired_state = {0};
    double past_x = 0, past_y = 0;
    double past_time = wb_robot_get_time();

    gains_pid_t gains_pid;
    gains_pid.kp_att_y  = 1;    gains_pid.kd_att_y  = 0.5;
    gains_pid.kp_att_rp = 0.5;  gains_pid.kd_att_rp = 0.1;
    gains_pid.kp_vel_xy = 2;    gains_pid.kd_vel_xy = 0.5;
    gains_pid.kp_z      = 10;   gains_pid.ki_z = 5; gains_pid.kd_z = 5;
    init_pid_attitude_fixed_height_controller();

    double height_desired = FLYING_ALTITUDE;
    motor_power_t motor_power;

    int was_in_box = 1, lap_count = 0;
    float speed_scalar = SPEED_SCALAR;

    printf("\n=== MANUAL CONTROLLER READY ===\n");
    printf("  W     = forward\n");
    printf("  A     = yaw left\n");
    printf("  D     = yaw right\n");
    printf("  Q     = rise\n");
    printf("  E     = descend\n");
    printf("  + / - = speed up / down (0.1 steps)\n");
    printf("  R     = reset simulation\n");
    printf("  Starting speed: %.1f\n\n", speed_scalar);

    while (wb_robot_step(timestep) != -1) {
        g_step++;
        const double dt      = wb_robot_get_time() - past_time;
        const float  simtime = (float)wb_robot_get_time();

        // ── Sensors ──────────────────────────────────────────────────────────
        actual_state.roll      = wb_inertial_unit_get_roll_pitch_yaw(imu)[0];
        actual_state.pitch     = wb_inertial_unit_get_roll_pitch_yaw(imu)[1];
        actual_state.yaw_rate  = wb_gyro_get_values(gyro)[2];
        actual_state.altitude  = wb_gps_get_values(gps)[2];
        double x = wb_gps_get_values(gps)[0];
        double y = wb_gps_get_values(gps)[1];
        double yaw    = wb_inertial_unit_get_roll_pitch_yaw(imu)[2];
        double cosyaw = cos(yaw);
        double sinyaw = sin(yaw);
        double vx_g   = (x - past_x) / dt;
        double vy_g   = (y - past_y) / dt;
        actual_state.vx =  vx_g * cosyaw + vy_g * sinyaw;
        actual_state.vy = -vx_g * sinyaw + vy_g * cosyaw;

        // ── Lap detection ─────────────────────────────────────────────────────
        int in_box = is_in_box(x, y);
        if (!was_in_box && in_box) {
            lap_count++;
            printf("  LAP %d  (t=%.1fs  speed=%.1f)\n", lap_count, (double)simtime, (double)speed_scalar);
        }
        was_in_box = in_box;

        // ── Keyboard input ────────────────────────────────────────────────────
        uint8_t fwd_cmd = 0, yaw_inc = 0, yaw_dec = 0, rst_cmd = 0;
        uint8_t rise = 0, descend = 0;
        int key;
        while ((key = wb_keyboard_get_key()) != -1) {
            switch (key) {
                case 'W':           fwd_cmd = 1;  break;
                case 'A':           yaw_inc = 1;  break;
                case 'D':           yaw_dec = 1;  break;
                case 'Q':           rise    = 1;  break;
                case 'E':           descend = 1;  break;
                case 'R':           rst_cmd = 1;  break;
                case '+': case '=':
                    speed_scalar = fminf(speed_scalar + 0.1f, 1.0f);
                    printf("  Speed: %.1f\n", (double)speed_scalar);
                    break;
                case '-': case '_':
                    speed_scalar = fmaxf(speed_scalar - 0.1f, 0.1f);
                    printf("  Speed: %.1f\n", (double)speed_scalar);
                    break;
            }
        }

        // ── Flight targets ────────────────────────────────────────────────────
        float yaw_scalar  = fmaxf(0.5f, 1.55f - 1.05f * (speed_scalar - 0.1f) / 0.9f);
        float fwd_desired = fwd_cmd ? speed_scalar : 0.0f;
        float yaw_desired = yaw_inc ? yaw_scalar : (yaw_dec ? -yaw_scalar : 0.0f);
        double height_diff = rise ? 0.1 : (descend ? -0.1 : 0.0);
        height_desired += height_diff * dt;

        desired_state.roll      = 0;
        desired_state.pitch     = 0;
        desired_state.vy        = 0;
        desired_state.vx        = (double)fwd_desired;
        desired_state.yaw_rate  = (double)yaw_desired;
        desired_state.altitude  = height_desired;

        pid_velocity_fixed_height_controller(actual_state, &desired_state, gains_pid, dt, &motor_power);

        // ── Motor output ──────────────────────────────────────────────────────
        float m1v = -(float)motor_power.m1;
        float m2v =  (float)motor_power.m2;
        float m3v = -(float)motor_power.m3;
        float m4v =  (float)motor_power.m4;
        wb_motor_set_velocity(m1_motor, (double)m1v);
        wb_motor_set_velocity(m2_motor, (double)m2v);
        wb_motor_set_velocity(m3_motor, (double)m3v);
        wb_motor_set_velocity(m4_motor, (double)m4v);

        // ── Send training packet ──────────────────────────────────────────────
        send_packet(camera, simtime,
                    fwd_cmd, yaw_inc, yaw_dec, rst_cmd,
                    fwd_desired, yaw_desired,
                    m1v, m2v, m3v, m4v,
                    speed_scalar);

        // ── Reset (packet already sent with rst=1) ────────────────────────────
        if (rst_cmd) {
            WbNodeRef self = wb_supervisor_node_get_self();
            wb_supervisor_simulation_reset();
            wb_supervisor_node_restart_controller(self);
            return 0;
        }

        // ── Periodic status ───────────────────────────────────────────────────
        if (g_step % 1000 == 0)
            printf("  t=%.0fs  steps=%d  speed=%.1f  laps=%d\n",
                   (double)simtime, g_step, (double)speed_scalar, lap_count);

        past_time = wb_robot_get_time();
        past_x = x;
        past_y = y;
    }

    if (g_socket >= 0) close(g_socket);
    wb_robot_cleanup();
    return 0;
}
