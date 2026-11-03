"""
boundary_core.py
════════════════
ISRO IROUC 2026 — Yellow Tape Border Detector  v5

Changes from v4
───────────────
  ✦ Nth-frame processing  — tune PROCESS_EVERY_N to reduce CPU
  ✦ No direct FC commands — publishes BoundaryResult to fc_interface
  ✦ fc_interface decides all behaviour (velocity, pivot, slow-down)
  ✦ Simulation mode kept — DroneFC sim prints commands
  ✦ INSIDE YELLOW → slow-down signal sent to fc_interface
  ✦ CENTRE TOUCHED → pivot signal sent to fc_interface
  ✦ All live-tune keys UNTOUCHED (U/u I/i O/o P/p ]/[ .,  =-)

Run standalone (sim):
    python3 boundary_core.py

Run on Pi with FC:
    python3 boundary_core.py --fc /dev/ttyUSB0 --baud 57600

Keys: Q=quit D=debug R=reset A=arm T=takeoff L=land S=save C=calib
Live-tune: U/u H_HIGH  I/i H_LOW  O/o S_LOW  P/p V_LOW
           ]/[ THRESH  .,  MIN_LEN  =- MAX_GAP
"""

import cv2
import numpy as np
import argparse
import time
import sys
import math
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False

# ── Calibration ───────────────────────────────────────────────────────────────
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "irouc_calib.json")

def load_calib() -> dict:
    defaults = {
        "H_LOW": 15, "H_HIGH": 40,
        "S_LOW": 90, "V_LOW":  70,
        "HOUGH_THRESH":  25,
        "HOUGH_MIN_LEN": 60,
        "HOUGH_MAX_GAP": 25,
        "HORIZ_TOL": 22,
        "VERT_TOL":  22,
        # ── compute cost knobs ───────────────────────────────────
        "PROCESS_EVERY_N": 3,        # process 1 in every N frames
        "DETECT_SCALE":    1.0,      # resize frame before detection (0.5 = half)
        "MAX_H_LINES":     6,        # cap horizontal lines kept
        "MAX_V_LINES":     6,        # cap vertical lines kept
    }
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                saved = json.load(f)
            defaults.update(saved)
            print(f"[CALIB] Loaded from {CALIB_FILE}")
        except Exception as e:
            print(f"[CALIB] Could not read: {e} — using defaults")
    else:
        print("[CALIB] No saved file — using defaults")
    return defaults

def save_calib(cfg: dict):
    try:
        with open(CALIB_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\n[CALIB] Saved → {CALIB_FILE}")
    except Exception as e:
        print(f"\n[CALIB] Save failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  PICAMERA2 SOURCE
# ═══════════════════════════════════════════════════════════════════

class PiCamSource:
    """
    Picamera2 capture source for Pi 5 + IMX296.
    Requests RGB888 from picamera2 (it handles debayering internally)
    then converts RGB→BGR for OpenCV.
    Falls back to V4L2 if picamera2 unavailable.
    """
    def __init__(self, width=1280, height=720, cam_index=1):
        self._use_picam = False
        self._cap       = None
        self._cam       = None
        try:
            from picamera2 import Picamera2
            self._cam = Picamera2(cam_index)
            # RGB888 — picamera2 handles ISP/debayer, gives us clean RGB
            cfg = self._cam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"})
            self._cam.configure(cfg)
            self._cam.start()
            time.sleep(1.0)   # give ISP time to settle
            # warm-up: discard first few frames
            for _ in range(5):
                self._cam.capture_array()
            self._use_picam = True
            print(f"[CAM] Picamera2 {width}x{height} started (IMX296)")
        except Exception as e:
            print(f"[CAM] Picamera2 failed ({e}) — trying V4L2")
            self._cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._use_picam = False

    def read(self):
        if self._use_picam:
            try:
                arr = self._cam.capture_array()
                if arr is None:
                    return False, None
                # Match downcam.py colour fix:
                # picamera2 RGB888 → COLOR_RGB2BGR → COLOR_BGR2RGB = raw array
                # net effect: use array directly as OpenCV frame
                bgr = cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return True, bgr
            except Exception as e:
                print(f"[CAM] capture error: {e}")
                return False, None
        else:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return False, None
            return True, frame

    def release(self):
        if self._use_picam:
            try: self._cam.stop(); self._cam.close()
            except Exception: pass
        else:
            try: self._cap.release()
            except Exception: pass


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

CAM_WIDTH  = 1280
CAM_HEIGHT = 720
MIN_LINE_FRAC       = 0.15
FRAMES_TO_CONFIRM   = 5
CONFIRMED_LOCK      = 45
DECAY_EVERY         = 4
CENTRE_MARGIN       = 20


# ═══════════════════════════════════════════════════════════════════
#  BOUNDARY RESULT  — this is what fc_interface reads
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BoundaryResult:
    """
    Published every processed frame. fc_interface reads this and
    decides velocity / pivot / slow-down behaviour.

    Signals
    -------
    inside_yellow   : drone is inside the tape rectangle → SLOW DOWN
    centre_touched  : frame centre is very close to tape centre → PIVOT TIME
    confirmed_sides : how many sides are confirmed (0-4)
    sides           : dict TOP/BOTTOM/LEFT/RIGHT → 0 or 1
    rect            : (x1,y1,x2,y2) of tape rectangle or None
    centre_err      : (err_x, err_y) pixels from tape centre to frame centre
    timestamp       : time.perf_counter() when this was computed
    """
    inside_yellow  : bool  = False
    centre_touched : bool  = False
    confirmed_sides: int   = 0
    sides          : dict  = field(default_factory=dict)
    rect           : Optional[Tuple] = None
    centre_err     : Optional[Tuple] = None
    timestamp      : float = field(default_factory=time.perf_counter)

    def valid(self) -> bool:
        return self.confirmed_sides >= 2


# ═══════════════════════════════════════════════════════════════════
#  COLOURS
# ═══════════════════════════════════════════════════════════════════

YELLOW     = (0,   220, 255)
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
GREEN      = (0,   220,   0)
CYAN       = (255, 220,   0)
ORANGE     = (0,   140, 255)
RED        = (0,     0, 220)

SIDE_COL = {
    "TOP":    (0,   255, 255),
    "BOTTOM": (0,   220,   0),
    "LEFT":   (220,   0, 220),
    "RIGHT":  (0,   150, 255),
}


# ═══════════════════════════════════════════════════════════════════
#  YELLOW MASK
# ═══════════════════════════════════════════════════════════════════

def make_yellow_mask(frame: np.ndarray, cfg: dict) -> np.ndarray:
    fh = frame.shape[0]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo   = np.array([cfg["H_LOW"],  cfg["S_LOW"], cfg["V_LOW"]])
    hi   = np.array([cfg["H_HIGH"], 255,          255         ])
    mask = cv2.inRange(hsv, lo, hi)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.dilate(mask, k3, iterations=1)
    return mask


# ═══════════════════════════════════════════════════════════════════
#  LINE
# ═══════════════════════════════════════════════════════════════════

class LineDir(Enum):
    H = auto()
    V = auto()
    D = auto()

class TapeLine:
    __slots__ = ('x1','y1','x2','y2','dir','length','angle','mx','my','fw','fh')

    def __init__(self, x1,y1,x2,y2, fw,fh, horiz_tol=22, vert_tol=22):
        self.x1,self.y1,self.x2,self.y2 = float(x1),float(y1),float(x2),float(y2)
        self.fw,self.fh = fw,fh
        dx,dy = x2-x1, y2-y1
        self.length = math.hypot(dx, dy)
        self.angle  = math.degrees(math.atan2(abs(dy), abs(dx)))
        self.mx = (x1+x2)/2
        self.my = (y1+y2)/2
        if   self.angle < horiz_tol:            self.dir = LineDir.H
        elif self.angle > (90 - vert_tol):      self.dir = LineDir.V
        else:                                   self.dir = LineDir.D

    def long_enough(self) -> bool:
        if self.dir == LineDir.H: return self.length >= self.fw * MIN_LINE_FRAC
        if self.dir == LineDir.V: return self.length >= self.fh * MIN_LINE_FRAC
        return False

    def side_label(self) -> Optional[str]:
        if self.dir == LineDir.H: return "TOP" if self.my < self.fh/2 else "BOTTOM"
        if self.dir == LineDir.V: return "LEFT" if self.mx < self.fw/2 else "RIGHT"
        return None


def detect_yellow_lines(frame: np.ndarray, cfg: dict,
                        scale: float = 1.0) -> Tuple[List[TapeLine], np.ndarray]:
    """
    Run yellow mask + Hough on frame.
    scale < 1.0 shrinks the frame before detection (cheaper).
    Lines are scaled back to original frame coords.
    """
    fh_orig, fw_orig = frame.shape[:2]

    if scale != 1.0:
        w = int(fw_orig * scale); h = int(fh_orig * scale)
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        small = frame

    fh, fw = small.shape[:2]
    mask = make_yellow_mask(small, cfg)

    segs = cv2.HoughLinesP(mask,
                           rho=1, theta=np.pi/180,
                           threshold=cfg["HOUGH_THRESH"],
                           minLineLength=cfg["HOUGH_MIN_LEN"],
                           maxLineGap=cfg["HOUGH_MAX_GAP"])

    # Scale mask back for display
    if scale != 1.0:
        mask = cv2.resize(mask, (fw_orig, fh_orig), interpolation=cv2.INTER_NEAREST)

    if segs is None:
        return [], mask

    inv = 1.0 / scale
    lines = []
    for seg in segs.reshape(-1, 4):
        x1,y1,x2,y2 = [v * inv for v in seg]
        ln = TapeLine(x1,y1,x2,y2, fw_orig, fh_orig,
                      horiz_tol=cfg["HORIZ_TOL"], vert_tol=cfg["VERT_TOL"])
        if ln.dir != LineDir.D and ln.long_enough():
            lines.append(ln)

    max_h = cfg.get("MAX_H_LINES", 6)
    max_v = cfg.get("MAX_V_LINES", 6)
    h_lines = sorted([l for l in lines if l.dir==LineDir.H], key=lambda l:-l.length)[:max_h]
    v_lines = sorted([l for l in lines if l.dir==LineDir.V], key=lambda l:-l.length)[:max_v]
    return h_lines + v_lines, mask


# ═══════════════════════════════════════════════════════════════════
#  SIDE STATE
# ═══════════════════════════════════════════════════════════════════

class SideState:
    def __init__(self, name: str):
        self.name           = name
        self.status         = 0
        self.build_count    = 0
        self.absent_count   = 0
        self.decay_tick     = 0
        self.pos            = 0.0
        self.best_line: Optional[TapeLine] = None

    def see(self, line: TapeLine):
        pos = line.my if line.dir == LineDir.H else line.mx
        self.pos = pos if self.build_count == 0 else 0.25*pos + 0.75*self.pos
        self.build_count  = min(self.build_count + 1, FRAMES_TO_CONFIRM + 5)
        self.absent_count = 0; self.decay_tick = 0; self.best_line = line
        if self.build_count >= FRAMES_TO_CONFIRM: self.status = 1

    def unseen(self):
        if self.status == 1:
            self.absent_count += 1
            if self.absent_count >= CONFIRMED_LOCK:
                self.status = 0; self.build_count = 0; self.best_line = None
        else:
            self.decay_tick += 1
            if self.decay_tick >= DECAY_EVERY:
                self.decay_tick = 0; self.build_count = max(0, self.build_count-1)

    def reset(self): self.__init__(self.name)

    @property
    def confirmed(self) -> bool: return self.status == 1


# ═══════════════════════════════════════════════════════════════════
#  BORDER MAP
# ═══════════════════════════════════════════════════════════════════

class BorderMap:
    def __init__(self):
        self.sides: Dict[str, SideState] = {
            n: SideState(n) for n in ("TOP","BOTTOM","LEFT","RIGHT")
        }
        self.centre_status = 0

    def reset(self):
        for s in self.sides.values(): s.reset()
        self.centre_status = 0

    @property
    def confirmed_count(self) -> int:
        return sum(1 for s in self.sides.values() if s.confirmed)

    def update(self, lines: List[TapeLine], fw: int, fh: int):
        visible = {}
        for name in ("TOP","BOTTOM","LEFT","RIGHT"):
            cands = [l for l in lines if l.side_label() == name]
            if cands: visible[name] = max(cands, key=lambda l: l.length)
        for name, state in self.sides.items():
            if name in visible: state.see(visible[name])
            else:               state.unseen()
        self._update_centre(fw, fh)

    def _update_centre(self, fw: int, fh: int):
        s = self.sides
        cx, cy = fw//2, fh//2
        m = CENTRE_MARGIN
        top    = int(s["TOP"].pos)    if s["TOP"].confirmed    else 0
        bottom = int(s["BOTTOM"].pos) if s["BOTTOM"].confirmed else fh
        left   = int(s["LEFT"].pos)   if s["LEFT"].confirmed   else 0
        right  = int(s["RIGHT"].pos)  if s["RIGHT"].confirmed  else fw
        if self.confirmed_count >= 2:
            self.centre_status = 2 if (left+m < cx < right-m and
                                        top+m  < cy < bottom-m) else 0
        else:
            self.centre_status = 0

    def get_rect(self, fw, fh) -> Optional[Tuple]:
        if self.confirmed_count < 2: return None
        s = self.sides
        return (int(s["LEFT"].pos)   if s["LEFT"].confirmed   else 0,
                int(s["TOP"].pos)    if s["TOP"].confirmed    else 0,
                int(s["RIGHT"].pos)  if s["RIGHT"].confirmed  else fw,
                int(s["BOTTOM"].pos) if s["BOTTOM"].confirmed else fh)

    def get_corners(self, fw, fh) -> Dict[str, Optional[Tuple]]:
        s = self.sides
        top_y  = int(s["TOP"].pos)    if s["TOP"].confirmed    else None
        bot_y  = int(s["BOTTOM"].pos) if s["BOTTOM"].confirmed else None
        left_x = int(s["LEFT"].pos)   if s["LEFT"].confirmed   else None
        right_x= int(s["RIGHT"].pos)  if s["RIGHT"].confirmed  else None
        return {
            "TL": (left_x,  top_y)  if left_x  is not None and top_y  is not None else None,
            "TR": (right_x, top_y)  if right_x is not None and top_y  is not None else None,
            "BL": (left_x,  bot_y)  if left_x  is not None and bot_y  is not None else None,
            "BR": (right_x, bot_y)  if right_x is not None and bot_y  is not None else None,
        }

    def build_result(self, fw: int, fh: int) -> BoundaryResult:
        """Build a BoundaryResult for fc_interface to consume."""
        rect = self.get_rect(fw, fh)
        centre_err = None
        inside_yellow = False
        centre_touched = False

        if rect is not None:
            x1, y1, x2, y2 = rect
            tape_cx = (x1 + x2) // 2
            tape_cy = (y1 + y2) // 2
            frame_cx = fw // 2
            frame_cy = fh // 2
            centre_err = (frame_cx - tape_cx, frame_cy - tape_cy)

            # Inside yellow — frame centre is inside the tape rectangle
            inside_yellow = (x1 < frame_cx < x2 and y1 < frame_cy < y2)

            # Centre touched — frame centre is very close to tape centre
            # This means drone is right at the edge centre → time to pivot
            centre_touched = (abs(centre_err[0]) < CENTRE_MARGIN and
                              abs(centre_err[1]) < CENTRE_MARGIN)

        return BoundaryResult(
            inside_yellow   = inside_yellow,
            centre_touched  = centre_touched,
            confirmed_sides = self.confirmed_count,
            sides           = {n: s.status for n, s in self.sides.items()},
            rect            = rect,
            centre_err      = centre_err,
        )


# ═══════════════════════════════════════════════════════════════════
#  FC INTERFACE BRIDGE
#  Uses fc_interface.py functions directly — that file is the
#  ONLY file that talks to the Pixhawk.
#
#  Speed constants (tuned to your drone):
#    SPEED_SEARCH  0.4  — survey/scan speed
#    SPEED_NORMAL  0.4  — navigating toward tape centre
#    SPEED_SLOW    0.15 — inside yellow, proximity to tape
#    SPEED_MAX     0.7  — hard cap, never exceed
#
#  Hold policy: ALWAYS soft hold (hover) — tight hold is disabled.
# ═══════════════════════════════════════════════════════════════════

SPEED_SEARCH  = 0.4    # yaw scan / survey
SPEED_NORMAL  = 0.4    # navigating toward tape centre
SPEED_SLOW    = 0.15   # inside yellow — proximity slow-down
SPEED_MAX     = 0.7    # absolute cap
CENTRE_THRESH = 35     # px deadband

YAW_RATE_SEARCH = 20.0  # deg/s slow rotate when no tape found


class BoundaryFC:
    """
    Reads BoundaryResult every processed frame and calls fc_interface.
    fc_interface.py is the source of truth — we never call MAVLink directly.

    Priority order per frame:
      1. centre_touched  → soft hover + log pivot signal
      2. inside_yellow   → slow retreat away from nearest tape side
      3. valid (>=2 sides confirmed) → navigate toward tape centre
      4. no tape         → slow yaw search
    """

    def __init__(self, sim: bool = True):
        self.sim = sim
        if not sim:
            try:
                import fc_interface as _fc
                self._fc = _fc
                print("[BoundaryFC] fc_interface loaded")
            except ImportError:
                print("[BoundaryFC] fc_interface not found — falling back to sim")
                self.sim = True
                self._fc = None
        else:
            self._fc = None
            print("[BoundaryFC] Simulation mode")

    def execute(self, result: BoundaryResult):
        """Call every processed frame."""

        # ── Priority 1: centre touched → pivot signal ─────────────
        if result.centre_touched:
            self._hover()
            print(f"\n[BOUNDARY] CENTRE TOUCHED — PIVOT SIGNAL  "
                  f"t={time.perf_counter():.3f}")
            return

        # ── Priority 2: inside yellow → slow retreat ───────────────
        if result.inside_yellow and result.rect is not None:
            self._retreat(result)
            return

        # ── Priority 3: tape found → navigate to centre ────────────
        if result.valid() and result.centre_err is not None:
            self._navigate(result)
            return

        # ── Priority 4: no tape → slow yaw search ─────────────────
        self._search()

    def _navigate(self, result: BoundaryResult):
        """Navigate toward tape rectangle centre at SPEED_NORMAL."""
        err_x, err_y = result.centre_err
        speed = min(SPEED_NORMAL, SPEED_MAX)
        vx, vy = 0.0, 0.0
        if abs(err_x) > abs(err_y):
            vy = math.copysign(speed, err_x)
        elif abs(err_y) > CENTRE_THRESH:
            vx = math.copysign(speed, err_y)
        # else: within deadband — hover
        self._send_vel(vx, vy)
        if self.sim:
            print(f"\r[SIM] NAV  vx={vx:+.2f} vy={vy:+.2f}  "
                  f"err=({err_x:+d},{err_y:+d})", end="", flush=True)

    def _retreat(self, result: BoundaryResult):
        """
        Inside yellow — back away from nearest confirmed tape at SPEED_SLOW.
        Never exceeds SPEED_SLOW so drone has time to react.
        """
        sides = result.sides
        speed = min(SPEED_SLOW, SPEED_MAX)
        vx, vy = 0.0, 0.0
        if   sides.get("TOP"):    vx =  speed   # away from top tape
        elif sides.get("BOTTOM"): vx = -speed   # away from bottom tape
        if   sides.get("LEFT"):   vy =  speed   # away from left tape
        elif sides.get("RIGHT"):  vy = -speed   # away from right tape
        self._send_vel(vx, vy)
        if self.sim:
            print(f"\r[SIM] INSIDE YELLOW — retreat  "
                  f"vx={vx:+.2f} vy={vy:+.2f}", end="", flush=True)

    def _search(self):
        """Slow yaw rotation to scan for tape."""
        if self.sim:
            print(f"\r[SIM] SEARCH — yaw {YAW_RATE_SEARCH}deg/s",
                  end="", flush=True)
            return
        self._fc.yaw(rate_deg_s=YAW_RATE_SEARCH, clockwise=True)

    def _hover(self):
        """Soft hover — always use hover(), never tight hold."""
        if self.sim:
            print("\r[SIM] HOVER (soft)", end="", flush=True)
            return
        self._fc.hover()

    def _send_vel(self, vx: float, vy: float):
        """Send velocity via fc_interface.send_velocity."""
        vx = max(-SPEED_MAX, min(SPEED_MAX, vx))
        vy = max(-SPEED_MAX, min(SPEED_MAX, vy))
        if self.sim: return
        self._fc.send_velocity(vx, vy, 0.0)

    # ── Passthrough helpers for main() keys ───────────────────────
    def arm(self):
        if self.sim: print("[SIM] ARM"); return
        self._fc.arm()

    def takeoff(self, alt=1.5):
        if self.sim: print(f"[SIM] TAKEOFF {alt}m"); return
        self._fc.takeoff(alt)

    def land(self):
        if self.sim: print("[SIM] LAND"); return
        self._fc.land()


# ═══════════════════════════════════════════════════════════════════
#  DISPLAY  (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════

def draw_frame(frame, lines, bmap, result: BoundaryResult,
               fps, yellow_mask, cfg) -> np.ndarray:
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    for ln in lines:
        col = SIDE_COL.get(ln.side_label(), YELLOW)
        cv2.line(vis,(int(ln.x1),int(ln.y1)),(int(ln.x2),int(ln.y2)),col,4,cv2.LINE_AA)
        cv2.circle(vis,(int(ln.mx),int(ln.my)),5,col,-1)

    for name, state in bmap.sides.items():
        if not state.confirmed: continue
        col = SIDE_COL[name]
        if name in ("TOP","BOTTOM"):
            y = int(state.pos)
            cv2.line(vis,(0,y),(fw,y),col,3)
        else:
            x = int(state.pos)
            cv2.line(vis,(x,0),(x,fh),col,3)

    rect = bmap.get_rect(fw, fh)
    if rect:
        x1,y1,x2,y2 = rect
        ov = vis.copy()
        col_fill = (0,60,0) if result.inside_yellow else (0,180,80)
        cv2.rectangle(ov,(x1,y1),(x2,y2),col_fill,2)
        alpha = 0.06 * bmap.confirmed_count
        cv2.addWeighted(ov, alpha, vis, 1-alpha, 0, vis)

    corners = bmap.get_corners(fw, fh)
    for label, pt in corners.items():
        if pt is None: continue
        cx,cy = int(np.clip(pt[0],18,fw-18)), int(np.clip(pt[1],18,fh-18))
        cv2.circle(vis,(cx,cy),18,YELLOW,-1)
        cv2.circle(vis,(cx,cy),18,BLACK,2)
        ox = -40 if "L" in label else 6
        oy = -6  if "T" in label else 28
        cv2.putText(vis,label,(cx+ox,cy+oy),cv2.FONT_HERSHEY_SIMPLEX,0.75,BLACK,3,cv2.LINE_AA)
        cv2.putText(vis,label,(cx+ox,cy+oy),cv2.FONT_HERSHEY_SIMPLEX,0.75,YELLOW,2,cv2.LINE_AA)

    frame_cx,frame_cy = fw//2,fh//2
    if bmap.centre_status == 2 and rect:
        x1,y1,x2,y2 = rect
        ov2 = vis.copy()
        cv2.rectangle(ov2,(x1,y1),(x2,y2),(0,100,0),-1)
        cv2.addWeighted(ov2,0.20,vis,0.80,0,vis)
        cv2.putText(vis,"2",(frame_cx-25,frame_cy+30),
                    cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,255,255),5,cv2.LINE_AA)

    cross_col = (0,255,255) if bmap.centre_status==2 else (180,180,180)
    cv2.drawMarker(vis,(frame_cx,frame_cy),cross_col,cv2.MARKER_CROSS,50,2,cv2.LINE_AA)

    # Status indicators for inside_yellow and centre_touched
    if result.inside_yellow:
        cv2.putText(vis,"INSIDE YELLOW — SLOWING",(fw//2-130,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2,cv2.LINE_AA)
    if result.centre_touched:
        cv2.putText(vis,"CENTRE TOUCHED — PIVOT",(fw//2-120,90),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2,cv2.LINE_AA)

    # Side status panel
    px,py,pw,ph = 10,fh-140,260,130
    bg = vis.copy()
    cv2.rectangle(bg,(px,py),(px+pw,py+ph),(0,0,0),-1)
    cv2.addWeighted(bg,0.65,vis,0.35,0,vis)
    cv2.rectangle(vis,(px,py),(px+pw,py+ph),(80,80,80),1)
    cv2.putText(vis,"BORDER SIDES",(px+8,py+22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
    for i,name in enumerate(("TOP","BOTTOM","LEFT","RIGHT")):
        state = bmap.sides[name]
        col   = SIDE_COL[name] if state.status else tuple(c//3 for c in SIDE_COL[name])
        ry    = py+45+i*22
        cv2.putText(vis,f"{name:<7}",(px+8,ry),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)
        vcol  = (0,255,0) if state.status else (80,80,80)
        cv2.putText(vis,f"= {state.status}",(px+85,ry),cv2.FONT_HERSHEY_SIMPLEX,0.55,vcol,2)
        if not state.status and state.build_count > 0:
            bw = int(80*state.build_count/FRAMES_TO_CONFIRM)
            cv2.rectangle(vis,(px+120,ry-12),(px+120+bw,ry-4),col,-1)
    cvcol = (0,255,255) if bmap.centre_status==2 else (80,80,80)
    cv2.putText(vis,f"CENTRE = {bmap.centre_status}",(px+8,py+ph-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,cvcol,2)

    # Compute cost panel (bottom-right)
    hr_x,hr_y,hr_w,hr_h = fw-310,fh-140,300,130
    bg2=vis.copy()
    cv2.rectangle(bg2,(hr_x,hr_y),(hr_x+hr_w,hr_y+hr_h),(0,0,0),-1)
    cv2.addWeighted(bg2,0.65,vis,0.35,0,vis)
    cv2.rectangle(vis,(hr_x,hr_y),(hr_x+hr_w,hr_y+hr_h),(80,80,80),1)
    cv2.putText(vis,"LIVE TUNE",(hr_x+6,hr_y+18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
    tune_rows = [
        (f"H_HIGH      {cfg['H_HIGH']:3d}", "U/u  +1/-1",  (0,220,255)),
        (f"H_LOW       {cfg['H_LOW']:3d}",  "I/i  +1/-1",  (0,180,255)),
        (f"S_LOW       {cfg['S_LOW']:3d}",  "O/o  +5/-5",  (180,255,0)),
        (f"V_LOW       {cfg['V_LOW']:3d}",  "P/p  +5/-5",  (255,200,0)),
        (f"THRESH      {cfg['HOUGH_THRESH']:3d}","]/[ +1/-1",(200,200,200)),
        (f"MIN_LEN     {cfg['HOUGH_MIN_LEN']:3d}",".,  +5/-5",(200,200,200)),
        (f"PROCESS_N   {cfg['PROCESS_EVERY_N']:3d}","N/n +1/-1",(180,180,255)),
        (f"SCALE       {cfg['DETECT_SCALE']:.2f}","M/m +.1/-.1",(180,180,255)),
    ]
    for i,(label,keys,col) in enumerate(tune_rows):
        ry = hr_y+28+i*13
        cv2.putText(vis,label,(hr_x+6,ry),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
        cv2.putText(vis,keys,(hr_x+190,ry),cv2.FONT_HERSHEY_SIMPLEX,0.36,(120,120,120),1)

    # Top HUD
    hb=vis.copy()
    cv2.rectangle(hb,(0,0),(fw,68),(0,0,0),-1)
    cv2.addWeighted(hb,0.55,vis,0.45,0,vis)
    cv2.putText(vis,f"FPS {fps:5.1f}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.80,GREEN,2)
    n_str = f"N={cfg['PROCESS_EVERY_N']} scale={cfg['DETECT_SCALE']:.1f}"
    cv2.putText(vis,n_str,(160,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,255),1)
    cv2.putText(vis,f"Confirmed:{bmap.confirmed_count}/4  Centre:{bmap.centre_status}  "
                f"Inside:{int(result.inside_yellow)}  Pivot:{int(result.centre_touched)}",
                (12,56),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,0),2)
    cv2.putText(vis,
        "Q=quit D=debug R=reset A=arm T=takeoff L=land S=save C=calib  "
        "| U/u Hhigh  I/i Hlow  O/o Slow  P/p Vlow  ]/[ Thresh  ., MinLen  =- Gap  "
        "N/n ProcessN  M/m Scale",
        (10,fh-8),cv2.FONT_HERSHEY_SIMPLEX,0.32,(120,120,120),1)
    return vis


def draw_debug(frame, yellow_mask, lines, bmap) -> np.ndarray:
    fh,fw = frame.shape[:2]
    pw,ph = fw//3, fh//4
    sm = cv2.resize(cv2.cvtColor(yellow_mask,cv2.COLOR_GRAY2BGR),(pw,ph))
    for ln in lines:
        col = SIDE_COL.get(ln.side_label(),(200,200,200))
        cv2.line(sm,(int(ln.x1*pw/fw),int(ln.y1*ph/fh)),
                    (int(ln.x2*pw/fw),int(ln.y2*ph/fh)),col,2)
    vis = frame.copy()
    vis[0:ph,fw-pw:fw] = sm
    cv2.rectangle(vis,(fw-pw,0),(fw,ph),(80,80,80),1)
    cv2.putText(vis,"Yellow mask",(fw-pw+4,14),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
    return vis


# ═══════════════════════════════════════════════════════════════════
#  CALIB WINDOW
# ═══════════════════════════════════════════════════════════════════

CALIB_WIN = "IROUC Calibration  [ S=save  C=close ]"

def create_calib_window(cfg):
    cv2.namedWindow(CALIB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CALIB_WIN, 500, 400)
    cv2.createTrackbar("H  LOW",       CALIB_WIN, cfg["H_LOW"],          179, lambda v:None)
    cv2.createTrackbar("H  HIGH",      CALIB_WIN, cfg["H_HIGH"],         179, lambda v:None)
    cv2.createTrackbar("S  LOW",       CALIB_WIN, cfg["S_LOW"],          255, lambda v:None)
    cv2.createTrackbar("V  LOW",       CALIB_WIN, cfg["V_LOW"],          255, lambda v:None)
    cv2.createTrackbar("Hough THRESH", CALIB_WIN, cfg["HOUGH_THRESH"],   120, lambda v:None)
    cv2.createTrackbar("Hough MIN LEN",CALIB_WIN, cfg["HOUGH_MIN_LEN"],  300, lambda v:None)
    cv2.createTrackbar("Hough MAX GAP",CALIB_WIN, cfg["HOUGH_MAX_GAP"],  100, lambda v:None)
    cv2.createTrackbar("HORIZ TOL",    CALIB_WIN, cfg["HORIZ_TOL"],       45, lambda v:None)
    cv2.createTrackbar("VERT  TOL",    CALIB_WIN, cfg["VERT_TOL"],        45, lambda v:None)
    cv2.createTrackbar("PROCESS N",    CALIB_WIN, cfg["PROCESS_EVERY_N"], 10, lambda v:None)

def read_calib_window(cfg):
    try:
        cfg["H_LOW"]           = cv2.getTrackbarPos("H  LOW",       CALIB_WIN)
        cfg["H_HIGH"]          = max(cfg["H_LOW"]+1, cv2.getTrackbarPos("H  HIGH", CALIB_WIN))
        cfg["S_LOW"]           = cv2.getTrackbarPos("S  LOW",       CALIB_WIN)
        cfg["V_LOW"]           = cv2.getTrackbarPos("V  LOW",       CALIB_WIN)
        cfg["HOUGH_THRESH"]    = max(1,cv2.getTrackbarPos("Hough THRESH",  CALIB_WIN))
        cfg["HOUGH_MIN_LEN"]   = max(1,cv2.getTrackbarPos("Hough MIN LEN", CALIB_WIN))
        cfg["HOUGH_MAX_GAP"]   = max(0,cv2.getTrackbarPos("Hough MAX GAP", CALIB_WIN))
        cfg["HORIZ_TOL"]       = max(1,cv2.getTrackbarPos("HORIZ TOL",     CALIB_WIN))
        cfg["VERT_TOL"]        = max(1,cv2.getTrackbarPos("VERT  TOL",     CALIB_WIN))
        cfg["PROCESS_EVERY_N"] = max(1,cv2.getTrackbarPos("PROCESS N",     CALIB_WIN))
    except cv2.error: pass
    return cfg

def sync_calib_window(cfg):
    try:
        cv2.setTrackbarPos("H  LOW",        CALIB_WIN, cfg["H_LOW"])
        cv2.setTrackbarPos("H  HIGH",       CALIB_WIN, cfg["H_HIGH"])
        cv2.setTrackbarPos("S  LOW",        CALIB_WIN, cfg["S_LOW"])
        cv2.setTrackbarPos("V  LOW",        CALIB_WIN, cfg["V_LOW"])
        cv2.setTrackbarPos("Hough THRESH",  CALIB_WIN, cfg["HOUGH_THRESH"])
        cv2.setTrackbarPos("Hough MIN LEN", CALIB_WIN, cfg["HOUGH_MIN_LEN"])
        cv2.setTrackbarPos("Hough MAX GAP", CALIB_WIN, cfg["HOUGH_MAX_GAP"])
        cv2.setTrackbarPos("PROCESS N",     CALIB_WIN, cfg["PROCESS_EVERY_N"])
    except cv2.error: pass



# ═══════════════════════════════════════════════════════════════════
#  WEB STREAMER  — serves annotated MJPEG on :8081
# ═══════════════════════════════════════════════════════════════════

class WebStreamer:
    def __init__(self, port=8081):
        import http.server as _hs
        import socketserver as _ss
        self._jpg     = None
        self._lock    = threading.Lock()
        self._running = True
        streamer      = self

        class _H(_hs.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path == '/':
                    body = (b'<html><body style="background:#111;margin:0">'
                            b'<img src="/stream" style="max-width:100%;display:block">'
                            b'</body></html>')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers(); self.wfile.write(body)
                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type',
                                     'multipart/x-mixed-replace; boundary=--jpgbnd')
                    self.send_header('Cache-Control', 'no-cache')
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
                else:
                    self.send_response(404); self.end_headers()

        class _S(_ss.ThreadingTCPServer):
            allow_reuse_address = True

        self._server = _S(('0.0.0.0', port), _H)
        threading.Thread(target=self._server.serve_forever,
                         name='web-stream', daemon=True).start()
        try:
            import socket as _sk
            s = _sk.socket(_sk.AF_INET, _sk.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close()
        except Exception: ip = 'localhost'
        print(f"[STREAM] http://{ip}:{port}/  — open in browser")

    def push(self, frame):
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self._lock: self._jpg = jpg.tobytes()

    def stop(self):
        self._running = False
        self._server.shutdown()


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IROUC 2026 Yellow Tape v5")
    parser.add_argument("--cam",         type=int, default=1)
    parser.add_argument("--fc",          type=str, default=None)
    parser.add_argument("--baud",        type=int, default=57600)
    parser.add_argument("--width",       type=int, default=CAM_WIDTH)
    parser.add_argument("--height",      type=int, default=CAM_HEIGHT)
    parser.add_argument("--stream-port", type=int, default=8081)
    parser.add_argument("--sim",         action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_calib()
    sim_mode = args.fc is None
    if args.fc and not sim_mode:
        try:
            import fc_interface
            fc_interface.connect(args.fc, args.baud)
        except Exception as e:
            print(f"[WARN] fc_interface connect failed: {e} — sim mode")
            sim_mode = True

    print("=" * 62)
    print("  ISRO IROUC 2026 — Yellow Tape Border Detector  v5")
    print(f"  FC: {args.fc or 'SIMULATION'}  |  PROCESS_N={cfg['PROCESS_EVERY_N']}")
    print("=" * 62)

    cap    = PiCamSource(width=args.width, height=args.height, cam_index=args.cam)
    web    = WebStreamer(port=args.stream_port)
    fc     = BoundaryFC(sim=sim_mode)
    bmap   = BorderMap()

    debug       = True
    frame_n     = 0
    last_result = BoundaryResult()
    last_lines  = []
    last_mask   = None
    _quit       = threading.Event()

    # ── Keyboard input via stdin (runs in terminal, no display needed) ────
    # Type single letters and press Enter: q d r a t l s U u I i O o P p etc
    def _kbd_loop():
        nonlocal debug, cfg, last_result
        nudged = False
        print("\nKeyboard (type letter + Enter):")
        print("  q=quit  d=debug  r=reset  a=arm  t=takeoff  l=land  s=save")
        print("  U/u=Hhigh  I/i=Hlow  O/o=Slow  P/p=Vlow")
        print("  ]/[=Thresh  ./,=MinLen  =-=Gap  N/n=ProcessN  M/m=Scale\n")
        while not _quit.is_set():
            try:
                key = input().strip()
            except EOFError:
                break
            if   key == 'q': _quit.set()
            elif key == 'd': debug = not debug; print(f"[debug={'ON' if debug else 'OFF'}]")
            elif key == 'r': bmap.reset(); last_result = BoundaryResult(); print("[RESET]")
            elif key == 'a': fc.arm()
            elif key == 't': fc.takeoff()
            elif key == 'l': fc.land()
            elif key == 's': save_calib(cfg)
            elif key == 'U': cfg["H_HIGH"] = min(179, cfg["H_HIGH"]+1);                nudged=True
            elif key == 'u': cfg["H_HIGH"] = max(cfg["H_LOW"]+1, cfg["H_HIGH"]-1);     nudged=True
            elif key == 'I': cfg["H_LOW"]  = min(cfg["H_HIGH"]-1, cfg["H_LOW"]+1);     nudged=True
            elif key == 'i': cfg["H_LOW"]  = max(0, cfg["H_LOW"]-1);                   nudged=True
            elif key == 'O': cfg["S_LOW"]  = min(255, cfg["S_LOW"]+5);   nudged=True
            elif key == 'o': cfg["S_LOW"]  = max(0,   cfg["S_LOW"]-5);   nudged=True
            elif key == 'P': cfg["V_LOW"]  = min(255, cfg["V_LOW"]+5);   nudged=True
            elif key == 'p': cfg["V_LOW"]  = max(0,   cfg["V_LOW"]-5);   nudged=True
            elif key == ']': cfg["HOUGH_THRESH"]  = min(120, cfg["HOUGH_THRESH"]+1);   nudged=True
            elif key == '[': cfg["HOUGH_THRESH"]  = max(1,   cfg["HOUGH_THRESH"]-1);   nudged=True
            elif key == '.': cfg["HOUGH_MIN_LEN"] = min(300, cfg["HOUGH_MIN_LEN"]+5);  nudged=True
            elif key == ',': cfg["HOUGH_MIN_LEN"] = max(1,   cfg["HOUGH_MIN_LEN"]-5);  nudged=True
            elif key == '=': cfg["HOUGH_MAX_GAP"] = min(100, cfg["HOUGH_MAX_GAP"]+1);  nudged=True
            elif key == '-': cfg["HOUGH_MAX_GAP"] = max(0,   cfg["HOUGH_MAX_GAP"]-1);  nudged=True
            elif key == 'N': cfg["PROCESS_EVERY_N"] = min(10, cfg["PROCESS_EVERY_N"]+1); nudged=True
            elif key == 'n': cfg["PROCESS_EVERY_N"] = max(1,  cfg["PROCESS_EVERY_N"]-1); nudged=True
            elif key == 'M': cfg["DETECT_SCALE"] = min(1.0, round(cfg["DETECT_SCALE"]+0.1,1)); nudged=True
            elif key == 'm': cfg["DETECT_SCALE"] = max(0.25,round(cfg["DETECT_SCALE"]-0.1,1)); nudged=True
            if nudged:
                print(f"  H:{cfg['H_LOW']}-{cfg['H_HIGH']} S:{cfg['S_LOW']} "
                      f"V:{cfg['V_LOW']} T:{cfg['HOUGH_THRESH']} "
                      f"L:{cfg['HOUGH_MIN_LEN']} G:{cfg['HOUGH_MAX_GAP']} "
                      f"N:{cfg['PROCESS_EVERY_N']} scale:{cfg['DETECT_SCALE']:.1f}")
                nudged = False

    kbd_thread = threading.Thread(target=_kbd_loop, name='kbd', daemon=True)
    kbd_thread.start()

    try:
        while not _quit.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01); continue

            t0 = time.perf_counter()
            fh, fw = frame.shape[:2]
            frame_n += 1

            n = max(1, cfg["PROCESS_EVERY_N"])
            if frame_n % n == 0:
                scale = max(0.25, min(1.0, float(cfg.get("DETECT_SCALE", 1.0))))
                lines, ymask = detect_yellow_lines(frame, cfg, scale=scale)
                bmap.update(lines, fw, fh)
                last_result = bmap.build_result(fw, fh)
                last_lines  = lines
                last_mask   = ymask
                fc.execute(last_result)

            fps = 1.0 / (time.perf_counter() - t0 + 1e-9)

            display = draw_frame(frame, last_lines, bmap,
                                 last_result, fps,
                                 last_mask if last_mask is not None
                                 else np.zeros((fh, fw), dtype=np.uint8),
                                 cfg)
            if debug and last_mask is not None:
                display = draw_debug(display, last_mask, last_lines, bmap)

            web.push(display)

            s = bmap.sides
            print(f"\r  T={s['TOP'].status} B={s['BOTTOM'].status} "
                  f"L={s['LEFT'].status} R={s['RIGHT'].status} "
                  f"CTR={bmap.centre_status} "
                  f"inside={int(last_result.inside_yellow)} "
                  f"pivot={int(last_result.centre_touched)} "
                  f"N={n} fps={fps:4.1f}",
                  end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        web.stop()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
