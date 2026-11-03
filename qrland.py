#!/usr/bin/env python3
"""
qr_land.py
══════════
ArUco + QR precision landing for Pi 5 + Picamera2.

Visual servoing pipeline
────────────────────────
  1. Picamera2 captures frame
  2. MarkerWorker detects ArUco / QR in background thread
  3. Pixel error (cx - fw/2, cy - fh/2) → PID → velocity command
  4. MAVLink SET_POSITION_TARGET_LOCAL_NED → FC

Landing sequence
────────────────
  SEARCH  → rotate slowly until marker found
  YAW     → correct yaw to match seed orientation
  CENTRE  → PID lateral servoing until centred (err < threshold)
  LAND    → MAV_CMD_NAV_LAND (precision mode)

Usage
─────
  python3 qr_land.py --seed seed_qr.png --port /dev/ttyUSB0 --baud 57600
  python3 qr_land.py --seed seed_qr.png --port udp:127.0.0.1:14550
"""

import argparse
import math
import os
import threading
import time
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings('ignore')

import cv2
import numpy as np

# ── Picamera2 ─────────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False
    print("[WARN] picamera2 not available — falling back to cv2.VideoCapture(0)")

# ── MAVLink ───────────────────────────────────────────────────────────────────
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("[WARN] pymavlink not installed — FC commands disabled")

# ── ArUco ─────────────────────────────────────────────────────────────────────
ARUCO_AVAILABLE = hasattr(cv2, 'aruco')

# ── WeChat QR (faster, optional) ──────────────────────────────────────────────
try:
    cv2.wechat_qrcode_WeChatQRCode()
    WECHAT_AVAILABLE = True
except Exception:
    WECHAT_AVAILABLE = False

DETECT_W = 640
DETECT_H = 360

ARUCO_DICTS = [
    (cv2.aruco.DICT_4X4_50,        "4x4_50"),
    (cv2.aruco.DICT_4X4_100,       "4x4_100"),
    (cv2.aruco.DICT_5X5_50,        "5x5_50"),
    (cv2.aruco.DICT_5X5_100,       "5x5_100"),
    (cv2.aruco.DICT_6X6_50,        "6x6_50"),
    (cv2.aruco.DICT_ARUCO_ORIGINAL,"ORIGINAL"),
] if ARUCO_AVAILABLE else []


# ─────────────────────────────────────────────────────────────────────────────
# PID
# ─────────────────────────────────────────────────────────────────────────────

class PID:
    def __init__(self, kp, ki, kd, limit=1.0):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.limit = limit
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_t   = None

    def update(self, error):
        now = time.time()
        dt  = (now - self._prev_t) if self._prev_t else 0.02
        dt  = max(dt, 1e-4)
        self._prev_t    = now
        self._integral  = max(-self.limit, min(self.limit,
                              self._integral + error * dt))
        deriv           = (error - self._prev_err) / dt
        self._prev_err  = error
        out = self.kp * error + self.ki * self._integral + self.kd * deriv
        return max(-self.limit, min(self.limit, out))

    def reset(self):
        self._integral = 0.0; self._prev_err = 0.0; self._prev_t = None


# ─────────────────────────────────────────────────────────────────────────────
# FC interface — visual servoing via MAVLink
# ─────────────────────────────────────────────────────────────────────────────

class FCInterface:
    """
    Thin MAVLink wrapper.
    send_velocity(vx, vy, vz) → SET_POSITION_TARGET_LOCAL_NED body frame
    send_yaw_rate(deg_s)      → body yaw rate
    land()                    → MAV_CMD_NAV_LAND precision mode
    """

    def __init__(self, connection_string: str, baud: int = 57600):
        self._conn_str = connection_string
        self._baud     = baud
        self.mav       = None
        self.connected = False
        self._lock     = threading.Lock()

    def connect(self) -> bool:
        if not MAVLINK_AVAILABLE:
            print("[FC] MAVLink not available"); return False
        try:
            print(f"[FC] Connecting → {self._conn_str}")
            self.mav = mavutil.mavlink_connection(
                self._conn_str, baud=self._baud)
            self.mav.wait_heartbeat(timeout=10)
            print(f"[FC] Heartbeat OK  sys={self.mav.target_system}")
            self.connected = True
            threading.Thread(target=self._hb_loop,
                             name='fc-heartbeat').start()
            return True
        except Exception as e:
            print(f"[FC] Connection failed: {e}")
            return False

    def _hb_loop(self):
        while self.connected:
            try:
                with self._lock:
                    self.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            except Exception:
                pass
            time.sleep(1.0)

    def send_velocity(self, vx: float, vy: float, vz: float):
        """
        Visual servoing command.
        vx = forward/backward  vy = left/right  vz = up(−)/down(+)
        Body-offset NED frame.
        """
        if not self.connected: return
        try:
            with self._lock:
                self.mav.mav.set_position_target_local_ned_send(
                    0,
                    self.mav.target_system, self.mav.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    0b0000111111000111,   # velocity only mask
                    0, 0, 0,
                    vx, vy, vz,
                    0, 0, 0, 0, 0)
        except Exception as e:
            print(f"[FC] send_velocity error: {e}")

    def send_yaw_rate(self, deg_s: float):
        """Yaw rate command in degrees/sec."""
        if not self.connected: return
        try:
            with self._lock:
                self.mav.mav.set_position_target_local_ned_send(
                    0,
                    self.mav.target_system, self.mav.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                    0b0000010111111111,   # yaw rate only
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, math.radians(deg_s))
        except Exception as e:
            print(f"[FC] send_yaw_rate error: {e}")

    def hover(self):
        self.send_velocity(0, 0, 0)

    def land(self, precision: int = 1):
        """
        MAV_CMD_NAV_LAND
        precision: 0=normal  1=opportunistic  2=required
        """
        if not self.connected: return
        try:
            with self._lock:
                self.mav.mav.command_long_send(
                    self.mav.target_system, self.mav.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_LAND,
                    0, 0, precision, 0, 0, 0, 0, 0)
            print("[FC] Precision land command sent")
        except Exception as e:
            print(f"[FC] land error: {e}")

    def disconnect(self):
        self.connected = False
        try:
            if self.mav: self.mav.close()
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Visual Servo Controller
# ─────────────────────────────────────────────────────────────────────────────

class VisualServo:
    """
    Converts pixel centring error → MAVLink velocity commands.

    Pipeline per frame:
      err_x, err_y = marker_centre - frame_centre   (pixels)
      norm_x = err_x / (frame_w / 2)                (−1 to +1)
      norm_y = err_y / (frame_h / 2)
      vx = pid_y.update(norm_y) * max_speed         (forward = reduce y err)
      vy = pid_x.update(norm_x) * max_speed         (lateral = reduce x err)
      → send_velocity(vx, vy, 0)

    Proximity slowdown: when marker fills >30% of frame width,
    max_speed is scaled down to crawl speed (min_speed).
    """

    # Landing state machine
    SEARCH  = "SEARCH"
    YAW     = "YAW"
    CENTRE  = "CENTRE"
    LAND    = "LAND"
    DONE    = "DONE"

    def __init__(self, fc: FCInterface,
                 max_speed: float = 0.3,
                 min_speed: float = 0.05,
                 centre_thresh_px: int = 25,
                 yaw_thresh_deg: float = 8.0,
                 yaw_rate_dps: float = 20.0,
                 centre_hold_frames: int = 10):

        self.fc                 = fc
        self.max_speed          = max_speed
        self.min_speed          = min_speed
        self.centre_thresh_px   = centre_thresh_px
        self.yaw_thresh_deg     = yaw_thresh_deg
        self.yaw_rate_dps       = yaw_rate_dps
        self.centre_hold_frames = centre_hold_frames

        self.pid_x   = PID(kp=0.5, ki=0.02, kd=0.08, limit=1.0)
        self.pid_y   = PID(kp=0.5, ki=0.02, kd=0.08, limit=1.0)
        self.pid_yaw = PID(kp=0.8, ki=0.01, kd=0.05, limit=45.0)

        self.state            = self.SEARCH
        self._centred_count   = 0
        self._landing_started = False
        self._cmd_cooldown    = 0.08
        self._last_cmd_t      = 0.0

    def update(self, marker_centre, frame_size,
               marker_size_px: float = 0.0,
               angle_diff: float = 0.0,
               angle_ok: bool = True):
        """
        Call every frame when marker is detected.

        Parameters
        ----------
        marker_centre   : (cx, cy) in pixels
        frame_size      : (fw, fh)
        marker_size_px  : mean side length of marker in pixels
        angle_diff      : yaw error in degrees (current − seed)
        angle_ok        : True when yaw is within tolerance
        """
        now = time.time()
        if now - self._last_cmd_t < self._cmd_cooldown:
            return
        self._last_cmd_t = now

        fw, fh = frame_size
        cx, cy = marker_centre

        # Pixel error
        err_x = cx - fw / 2.0
        err_y = cy - fh / 2.0

        # Normalised error (−1 to +1)
        norm_x = err_x / (fw / 2.0)
        norm_y = err_y / (fh / 2.0)

        # Proximity scale — slow down as marker gets larger
        prox   = self._proximity_scale(marker_size_px, fw)
        speed  = self.min_speed + (self.max_speed - self.min_speed) * prox

        # ── State machine ─────────────────────────────────────────
        if self.state == self.SEARCH:
            # Marker just found — move to YAW or CENTRE
            self.state = self.YAW if not angle_ok else self.CENTRE
            self.pid_x.reset(); self.pid_y.reset(); self.pid_yaw.reset()

        if self.state == self.YAW:
            if not angle_ok:
                yr = self.pid_yaw.update(angle_diff)
                self.fc.send_yaw_rate(yr)
                return
            else:
                self.state = self.CENTRE

        if self.state == self.CENTRE:
            vx = self.pid_y.update(norm_y) * speed
            vy = self.pid_x.update(norm_x) * speed
            self.fc.send_velocity(vx, vy, 0.0)

            centred = (abs(err_x) < self.centre_thresh_px and
                       abs(err_y) < self.centre_thresh_px)
            if centred:
                self._centred_count += 1
            else:
                self._centred_count  = 0

            if self._centred_count >= self.centre_hold_frames:
                self.state = self.LAND

        if self.state == self.LAND and not self._landing_started:
            self._landing_started = True
            self.fc.hover()
            time.sleep(0.3)
            self.fc.land(precision=1)
            self.state = self.DONE
            print("[SERVO] LAND command sent")

    def on_lost(self):
        """Called when marker is not detected."""
        if self.state in (self.CENTRE, self.YAW):
            self.fc.hover()
        if self.state == self.SEARCH:
            self.fc.send_yaw_rate(self.yaw_rate_dps)
        self._centred_count = 0

    def reset(self):
        self.state            = self.SEARCH
        self._centred_count   = 0
        self._landing_started = False
        self.pid_x.reset(); self.pid_y.reset(); self.pid_yaw.reset()

    @staticmethod
    def _proximity_scale(marker_size_px: float, frame_w: int) -> float:
        if marker_size_px <= 0 or frame_w <= 0:
            return 1.0
        frac  = marker_size_px / frame_w
        scale = 1.0 - (frac / 0.35) * 0.85
        return float(max(0.05, min(1.0, scale)))


# ─────────────────────────────────────────────────────────────────────────────
# Picamera2 source
# ─────────────────────────────────────────────────────────────────────────────

class PiCamSource:
    def __init__(self, width=1280, height=720, cam_index=0):
        if PICAM_AVAILABLE:
            self._cam = Picamera2(cam_index)
            cfg = self._cam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"})
            self._cam.configure(cfg)
            self._cam.start()
            time.sleep(0.5)
            self._use_picam = True
            print(f"[CAM] Picamera2 {width}x{height} started")
        else:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._use_picam = False
            print("[CAM] Fallback cv2.VideoCapture(0)")

        self._frame = None
        self._lock  = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, name='cam', daemon=True).start()

    def _loop(self):
        while self._running:
            if self._use_picam:
                try:
                    arr = self._cam.capture_array()
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    with self._lock: self._frame = bgr
                except Exception as e:
                    print(f"[CAM] capture error: {e}")
                    time.sleep(0.02)
            else:
                ret, frame = self._cap.read()
                if ret:
                    with self._lock: self._frame = frame
                else:
                    time.sleep(0.02)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._use_picam:
            try: self._cam.stop(); self._cam.close()
            except Exception: pass
        else:
            try: self._cap.release()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Seed matcher
# ─────────────────────────────────────────────────────────────────────────────

class SeedMatcher:
    def __init__(self):
        self.seed_data      = None
        self.seed_loaded    = False
        self.seed_aruco_id  = None
        self.seed_dict_id   = None
        self.seed_dict_name = ""
        self._ref_angle     = None
        self.angle_tolerance = 10.0

    def load(self, filepath: str) -> tuple:
        img = cv2.imread(filepath)
        if img is None:
            return False, f"Cannot read: {filepath}"
        img  = cv2.resize(img, (400, 400))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if ARUCO_AVAILABLE:
            params = cv2.aruco.DetectorParameters()
            for dict_id, dict_name in ARUCO_DICTS:
                d   = cv2.aruco.getPredefinedDictionary(dict_id)
                det = cv2.aruco.ArucoDetector(d, params)
                for src in [gray, cv2.equalizeHist(gray)]:
                    corners, ids, _ = det.detectMarkers(src)
                    if ids is not None and len(ids) > 0:
                        mid = int(ids[0][0])
                        self.seed_data      = f"ARUCO:{mid}"
                        self.seed_aruco_id  = mid
                        self.seed_dict_id   = dict_id
                        self.seed_dict_name = dict_name
                        self.seed_loaded    = True
                        c4 = corners[0].reshape(4, 2).astype(np.float32)
                        self._ref_angle = self._angle(c4)
                        print(f"[SEED] ArUco ID {mid} ({dict_name})  "
                              f"ref_angle={self._ref_angle:.1f}deg")
                        return True, f"ArUco ID {mid} dict={dict_name}"

        det = cv2.QRCodeDetector()
        for src in [img, gray]:
            data, pts, _ = det.detectAndDecode(src)
            if data and pts is not None:
                self.seed_data   = data
                self.seed_loaded = True
                self._ref_angle  = self._angle(pts[0].astype(np.float32))
                print(f"[SEED] QR '{data[:40]}'  ref_angle={self._ref_angle:.1f}deg")
                return True, f"QR: {data[:60]}"

        return False, "No ArUco / QR found in seed"

    def match(self, label: str) -> bool:
        if not self.seed_loaded or not label:
            return False
        return label == self.seed_data

    def orientation(self, corners) -> dict:
        """Returns angle_diff and whether yaw is OK."""
        if self._ref_angle is None or corners is None:
            return {'angle_diff': 0.0, 'angle_ok': True}
        curr = self._angle(corners.astype(np.float32))
        diff = (curr - self._ref_angle + 180.0) % 360.0 - 180.0
        return {
            'angle_diff': diff,
            'angle_ok'  : abs(diff) <= self.angle_tolerance,
        }

    @staticmethod
    def _angle(c):
        e = c[1] - c[0]
        return math.degrees(math.atan2(float(e[1]), float(e[0])))


# ─────────────────────────────────────────────────────────────────────────────
# Marker detection worker
# ─────────────────────────────────────────────────────────────────────────────

class MarkerWorker:
    def __init__(self, seed_matcher: SeedMatcher):
        self.sm           = seed_matcher
        self.qr_det       = cv2.QRCodeDetector()
        self._clahe       = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self._in_frame    = None
        self._result      = None   # (label, corners, angle_diff, angle_ok, marker_size_px)
        self._lock        = threading.Lock()
        self.running      = True
        self._locked_det  = None

        if ARUCO_AVAILABLE:
            params = cv2.aruco.DetectorParameters()
            params.minMarkerPerimeterRate    = 0.015
            params.cornerRefinementMethod   = cv2.aruco.CORNER_REFINE_SUBPIX
            self._aruco_params = params
            self._aruco_dets   = [
                (did, nm, cv2.aruco.ArucoDetector(
                    cv2.aruco.getPredefinedDictionary(did), params))
                for did, nm in ARUCO_DICTS
            ]

        self._wechat = None
        if WECHAT_AVAILABLE:
            try: self._wechat = cv2.wechat_qrcode_WeChatQRCode()
            except Exception: pass

        threading.Thread(target=self._run, name='marker-worker',
                         daemon=True).start()

    def lock_dict(self, dict_id):
        if not ARUCO_AVAILABLE or dict_id is None: return
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        self._locked_det = cv2.aruco.ArucoDetector(d, self._aruco_params)
        print(f"[WORKER] Locked to ArUco dict {dict_id}")

    def submit(self, frame):
        with self._lock:
            if self._in_frame is None:
                self._in_frame = frame.copy()

    def get(self):
        with self._lock:
            return self._result

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        sx, sy = w / DETECT_W, h / DETECT_H
        small = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
        return small, sx, sy

    def _scale_pts(self, pts, sx, sy):
        c = pts.copy()
        c[:, 0] *= sx; c[:, 1] *= sy
        return c

    def _detect_aruco(self, gray_small, sx, sy):
        if not ARUCO_AVAILABLE: return None, None
        imgs = [gray_small, self._clahe.apply(gray_small)]
        dets = [(None, None, self._locked_det)] if self._locked_det else self._aruco_dets
        for img in imgs:
            for _, _, det in dets:
                try:
                    corners, ids, _ = det.detectMarkers(img)
                    if ids is not None and len(ids) > 0:
                        c4 = corners[0].reshape(4, 2).astype(np.float32)
                        return f"ARUCO:{int(ids[0][0])}", self._scale_pts(c4, sx, sy)
                except Exception: continue
        return None, None

    def _detect_qr(self, gray_small, sx, sy):
        if self._wechat:
            try:
                texts, pts_list = self._wechat.detectAndDecode(gray_small)
                if texts and pts_list:
                    pts = np.array(pts_list[0], dtype=np.float32)
                    if pts.shape[0] == 4:
                        return texts[0], self._scale_pts(pts, sx, sy)
            except Exception: pass
        clahe = self._clahe.apply(gray_small)
        for src in [gray_small, clahe, cv2.bitwise_not(gray_small)]:
            try:
                d, p, _ = self.qr_det.detectAndDecode(src)
                if p is not None and len(p) > 0 and p[0].shape[0] == 4:
                    return (d or ""), self._scale_pts(p[0].astype(np.float32), sx, sy)
            except Exception: continue
        return None, None

    def _run(self):
        while self.running:
            with self._lock: frame = self._in_frame
            if frame is None: time.sleep(0.001); continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
                   if len(frame.shape) == 3 else frame
            small, sx, sy = self._downscale(gray)

            label, corners = self._detect_aruco(small, sx, sy)
            if corners is None:
                label, corners = self._detect_qr(small, sx, sy)

            result = None
            if corners is not None and label and self.sm.match(label):
                orient = self.sm.orientation(corners)
                sides  = [np.linalg.norm(corners[(i+1)%4] - corners[i])
                          for i in range(4)]
                msize  = float(np.mean(sides))
                result = (label, corners,
                          orient['angle_diff'], orient['angle_ok'], msize)

            with self._lock:
                self._result   = result
                self._in_frame = None

    def stop(self): self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame, corners, servo_state, angle_diff, centred):
    fh, fw = frame.shape[:2]
    p      = corners.astype(np.int32)
    col    = (0,255,0) if centred else (0,165,255)

    for i in range(4):
        cv2.line(frame, tuple(p[i]), tuple(p[(i+1)%4]), col, 2)
    cx = int(np.mean(p[:,0])); cy = int(np.mean(p[:,1]))
    cv2.drawMarker(frame, (cx,cy), col, cv2.MARKER_CROSS, 30, 2)
    cv2.line(frame, (fw//2, fh//2), (cx,cy), (200,200,0), 1)
    cv2.drawMarker(frame, (fw//2, fh//2),
                   (255,255,255), cv2.MARKER_CROSS, 20, 1)

    err_x = cx - fw//2; err_y = cy - fh//2
    cv2.putText(frame,
        f"STATE:{servo_state}  err:({err_x:+d},{err_y:+d})px  yaw:{angle_diff:+.1f}deg",
        (10, fh-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(frame,
        f"CENTRED: {'YES' if centred else 'NO'}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0,255,0) if centred else (0,100,255), 2)


# ─────────────────────────────────────────────────────────────────────────────
# WebStreamer — view on browser
# ─────────────────────────────────────────────────────────────────────────────

class WebStreamer:
    def __init__(self, port=8084):
        import http.server, socketserver, socket as _socket
        self._jpg     = None
        self._lock    = threading.Lock()
        self._running = True
        streamer      = self

        class _H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path == '/':
                    body = (b'<html><body style="background:#111">'
                            b'<img src="/stream" style="max-width:100%"></body></html>')
                    self.send_response(200)
                    self.send_header('Content-Type','text/html')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers(); self.wfile.write(body)
                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type',
                                     'multipart/x-mixed-replace; boundary=--jpgbnd')
                    self.end_headers()
                    try:
                        while streamer._running:
                            with streamer._lock: jpg = streamer._jpg
                            if jpg is None: time.sleep(0.02); continue
                            self.wfile.write(
                                b'--jpgbnd\r\nContent-Type: image/jpeg\r\n'
                                + f'Content-Length: {len(jpg)}\r\n\r\n'.encode()
                                + jpg + b'\r\n')
                            time.sleep(0.033)
                    except (BrokenPipeError, ConnectionResetError): pass
                else: self.send_response(404); self.end_headers()

        class _S(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        self._server = _S(('0.0.0.0', port), _H)
        threading.Thread(target=self._server.serve_forever,
                         name='web-stream', daemon=True).start()
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(('8.8.8.8',80)); ip = s.getsockname()[0]; s.close()
        except Exception: ip = 'localhost'
        print(f"[STREAM] http://{ip}:{port}/")

    def push(self, frame):
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self._lock: self._jpg = jpg.tobytes()

    def stop(self):
        self._running = False
        self._server.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="QR/ArUco precision landing — Pi5")
    ap.add_argument('--seed',   required=True,          help='Seed image path')
    ap.add_argument('--port',   default='/dev/ttyUSB0', help='FC port or udp:ip:port')
    ap.add_argument('--baud',   type=int, default=57600)
    ap.add_argument('--width',  type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--cam',    type=int, default=0)
    ap.add_argument('--stream-port', type=int, default=8084)
    ap.add_argument('--no-fc',  action='store_true', help='Vision only, no FC')
    ap.add_argument('--max-speed', type=float, default=0.3)
    ap.add_argument('--centre-thresh', type=int, default=25)
    args = ap.parse_args()

    # ── Seed ────────────────────────────────────────────────────────────
    sm = SeedMatcher()
    ok, msg = sm.load(args.seed)
    if not ok:
        print(f"[ERROR] Seed load failed: {msg}"); return
    print(f"[OK] Seed: {msg}")

    # ── FC ──────────────────────────────────────────────────────────────
    fc = FCInterface(args.port, args.baud)
    if not args.no_fc:
        fc.connect()

    # ── Visual servo ────────────────────────────────────────────────────
    servo = VisualServo(fc,
                        max_speed       = args.max_speed,
                        centre_thresh_px= args.centre_thresh)

    # ── Camera ──────────────────────────────────────────────────────────
    cam    = PiCamSource(args.width, args.height, args.cam)
    worker = MarkerWorker(sm)
    if sm.seed_dict_id is not None:
        worker.lock_dict(sm.seed_dict_id)

    web    = WebStreamer(port=args.stream_port)

    lost_frames = 0
    LOSE_AFTER  = 20
    active      = True   # set False to disable servo commands

    print("\n[RUN] Press Ctrl+C to stop")
    print(f"      Stream: http://<pi-ip>:{args.stream_port}/\n")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01); continue

            worker.submit(frame)
            result = worker.get()

            disp = frame.copy()

            if result is not None:
                label, corners, angle_diff, angle_ok, msize = result
                lost_frames = 0

                fh, fw = frame.shape[:2]
                cx = int(np.mean(corners[:,0]))
                cy = int(np.mean(corners[:,1]))

                centred = (abs(cx - fw//2) < args.centre_thresh and
                           abs(cy - fh//2) < args.centre_thresh)

                if active and not args.no_fc:
                    servo.update(
                        marker_centre   = (cx, cy),
                        frame_size      = (fw, fh),
                        marker_size_px  = msize,
                        angle_diff      = angle_diff,
                        angle_ok        = angle_ok,
                    )

                draw_overlay(disp, corners, servo.state, angle_diff, centred)

            else:
                lost_frames += 1
                if lost_frames >= LOSE_AFTER:
                    if active and not args.no_fc:
                        servo.on_lost()
                cv2.putText(disp, f"SEARCHING... (state:{servo.state})",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,165,255), 2)

            if servo.state == VisualServo.DONE:
                cv2.putText(disp, "LANDED", (disp.shape[1]//2 - 60, disp.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            web.push(disp)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        if not args.no_fc:
            fc.hover()
            time.sleep(0.3)
            fc.disconnect()
        worker.stop()
        cam.stop()
        web.stop()


if __name__ == '__main__':
    main()
