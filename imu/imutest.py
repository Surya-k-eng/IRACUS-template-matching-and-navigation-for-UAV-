#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
import matplotlib.pyplot as plt

# -----------------------------
# EKF CORE
# -----------------------------
def quat_mul(q1, q2):
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_to_R(q):
    x,y,z,w = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def normalize(q):
    return q / np.linalg.norm(q)

class EKF_VIO:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([0,0,0,1])
        self.P = np.eye(10) * 0.1
        self.g = np.array([0,0,9.81])

    def predict(self, accel, gyro, dt):
        R = quat_to_R(self.q)

        a_world = R @ accel - self.g

        self.p += self.v * dt + 0.5 * a_world * dt**2
        self.v += a_world * dt

        omega = np.array([gyro[0], gyro[1], gyro[2], 0])
        dq = 0.5 * quat_mul(self.q, omega) * dt
        self.q += dq
        self.q = normalize(self.q)

        self.P += np.eye(10) * 0.001

    def update_vo(self, t_vo):
        z = t_vo.flatten()

        H = np.zeros((3,10))
        H[:,0:3] = np.eye(3)

        Rm = np.eye(3) * 0.05

        y = z - self.p
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)

        dx = K @ y

        self.p += dx[0:3]
        self.v += dx[3:6]

        self.P = (np.eye(10) - K @ H) @ self.P


# -----------------------------
# VISUAL ODOMETRY
# -----------------------------
class SimpleVO:
    def __init__(self):
        self.K = np.array([[600,0,320],[0,600,240],[0,0,1]], dtype=np.float32)
        self.orb = cv2.ORB_create(1000)
        self.prev_gray = None
        self.prev_pts = None

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            kp = self.orb.detect(gray, None)
            self.prev_pts = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1,1,2)
            self.prev_gray = gray
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None)

        good_old = self.prev_pts[status==1]
        good_new = next_pts[status==1]

        if len(good_old) < 8:
            return None

        E, _ = cv2.findEssentialMat(good_old, good_new, self.K)

        if E is None:
            return None

        _, R, t, _ = cv2.recoverPose(E, good_old, good_new, self.K)

        self.prev_pts = good_new.reshape(-1,1,2)
        self.prev_gray = gray

        return t  # translation (scale unknown)


# -----------------------------
# MAIN SYSTEM
# -----------------------------
def run(bag_path, start=1900, end=2400):
    bag_path = Path(bag_path)

    images = []
    imus = []

    with AnyReader([bag_path]) as reader:
        image_conn = None
        imu_conn = None

        for c in reader.connections:
            if c.msgtype == "sensor_msgs/msg/Image":
                image_conn = c
            elif c.msgtype == "sensor_msgs/msg/Imu":
                imu_conn = c

        for conn, t, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)

            if conn == image_conn:
                images.append((t, message_to_cvimage(msg, "bgr8")))
            elif conn == imu_conn:
                imus.append((t,
                             msg.linear_acceleration.x,
                             msg.linear_acceleration.y,
                             msg.linear_acceleration.z,
                             msg.angular_velocity.x,
                             msg.angular_velocity.y,
                             msg.angular_velocity.z))

    images = images[start:end]

    print(f"Frames: {len(images)} | IMU: {len(imus)}")

    ekf = EKF_VIO()
    vo = SimpleVO()

    imu_idx = 0
    last_imu_t = None

    trajectory = []

    # 3D plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'g-')

    for i, (t_img, frame) in enumerate(images):

        # ---------------- IMU PREDICT ----------------
        while imu_idx < len(imus) and imus[imu_idx][0] <= t_img:
            t, ax_i, ay_i, az_i, gx, gy, gz = imus[imu_idx]

            if last_imu_t is not None:
                dt = (t - last_imu_t) / 1e9

                ekf.predict(
                    np.array([ax_i, ay_i, az_i]),
                    np.array([gx, gy, gz]),
                    dt
                )

            last_imu_t = t
            imu_idx += 1

        # ---------------- VO UPDATE ----------------
        vo_t = vo.process(frame.copy())

        if vo_t is not None:
            ekf.update_vo(vo_t)

        # ---------------- STORE ----------------
        trajectory.append(ekf.p.copy())

        # ---------------- CAMERA ----------------
        frame = frame.copy()
        cv2.putText(frame, f"Frame: {start+i}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Camera", frame)

        # ---------------- 3D ----------------
        traj = np.array(trajectory)
        line.set_data(traj[:,0], traj[:,1])
        line.set_3d_properties(traj[:,2])

        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_zlim(-5,5)

        plt.draw()
        plt.pause(0.001)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


# -----------------------------
if __name__ == "__main__":
    import sys
    bag = sys.argv[1] if len(sys.argv)>1 else "/home/daniel/Downloads/flight_dataset2.bag"
    run(bag)