"""
centering.py
════════════
Visual centering module for ASCEND.

Flow
────
  1. ORB locks on target → bounding box snapshot taken from frame
  2. centering.py receives snapshot + live frames
  3. PD controller centres drone over target using pixel error
  4. If lock holds for CONFIRM_SECS after centring → CONFIRMED
  5. On confirm:
       - Snapshot saved to acquired_seeds/<seed_name>_<timestamp>.jpg (Pi)
       - Acquisition signal written to /tmp/target_acquired.json
       - orb_offload_client.py (PC) reads signal → saves HD frame to
         acquired_targets/<seed_name>_<timestamp>.jpg

Triggering
──────────
  From autopilot REPL:
      center          → start centering on current ORB lock
      center stop     → abort centering
      center status   → show centering state

  Programmatically:
      from centering import Centerer
      c = Centerer(fc_interface=fc, seed_name='seed1')
      c.start(snapshot_bgr, frame_size)
      # feed frames:
      c.update(lock_box, frame_w, frame_h)
      # check result:
      if c.confirmed: ...

Folder structure (Pi)
─────────────────────
  seeds/                   ← reference seeds (input)
  acquired_seeds/          ← confirmed target snapshots
  acquired_targets/        ← written by PC via signal
"""

import json
import logging
import math
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("centering")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CENTRE_THRESH_PX  = 25      # px — centred when err < this
CONFIRM_SECS      = 4.0     # seconds to hold centred before confirming
CENTRE_SPEED_MAX  = 0.3     # m/s max lateral speed
CENTRE_SPEED_MIN  = 0.05    # m/s min lateral speed
KP                = 0.004   # proportional gain
KD                = 0.001   # derivative gain
ACQUIRED_SEEDS_DIR   = Path('acquired_seeds')
ACQUIRED_TARGETS_DIR = Path('acquired_targets')
ACQUIRE_SIGNAL_FILE  = Path('/tmp/target_acquired.json')


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class CenterState:
    IDLE      = 'IDLE'
    CENTERING = 'CENTERING'
    CENTRED   = 'CENTRED'
    CONFIRMED = 'CONFIRMED'
    FAILED    = 'FAILED'


# ─────────────────────────────────────────────────────────────────────────────
# Centerer
# ─────────────────────────────────────────────────────────────────────────────

class Centerer:
    """
    Visual servoing centerer.

    Usage
    -----
        c = Centerer(send_velocity_fn, seed_name='seed1')
        c.start(snapshot_bgr, frame_w, frame_h)

        # Each frame with new lock_box from ORB:
        c.update(lock_box=(x1,y1,x2,y2), frame_w=640, frame_h=480)

        if c.state == CenterState.CONFIRMED:
            print("Target acquired!")
    """

    def __init__(self, send_velocity_fn, hover_fn,
                 seed_name: str = 'target',
                 centre_thresh: int = CENTRE_THRESH_PX,
                 confirm_secs: float = CONFIRM_SECS):

        self._send_vel    = send_velocity_fn
        self._hover       = hover_fn
        self.seed_name    = seed_name
        self.centre_thresh= centre_thresh
        self.confirm_secs = confirm_secs

        self.state        = CenterState.IDLE
        self.snapshot     = None    # BGR crop of target at lock time
        self.confirmed    = False

        self._centred_since = None
        self._prev_err_x    = 0
        self._prev_err_y    = 0
        self._lock          = threading.Lock()
        self._running       = False
        self._last_update   = 0.0

        ACQUIRED_SEEDS_DIR.mkdir(exist_ok=True)
        ACQUIRED_TARGETS_DIR.mkdir(exist_ok=True)

    def start(self, snapshot_bgr: np.ndarray,
              frame_w: int, frame_h: int):
        """
        Begin centering. snapshot_bgr is the crop inside the bounding box
        at the moment of ORB lock — saved as reference.
        """
        with self._lock:
            self.snapshot      = snapshot_bgr.copy()
            self.state         = CenterState.CENTERING
            self.confirmed     = False
            self._centred_since= None
            self._prev_err_x   = 0
            self._prev_err_y   = 0
            self._running      = True
            self._frame_w      = frame_w
            self._frame_h      = frame_h
        log.info(f"[Centerer] started  seed={self.seed_name}  "
                 f"frame={frame_w}x{frame_h}")

    def update(self, lock_box: Optional[Tuple],
               frame_w: int, frame_h: int) -> str:
        """
        Call every frame with latest ORB lock_box (or None if lost).
        Returns current state string.
        Sends velocity commands to FC to centre.
        """
        with self._lock:
            if not self._running:
                return self.state

        if lock_box is None:
            # Lost lock during centering
            log.warning("[Centerer] lock lost — aborting")
            self._hover()
            with self._lock:
                self.state    = CenterState.FAILED
                self._running = False
            return self.state

        x1, y1, x2, y2 = lock_box
        bcx = (x1 + x2) / 2.0
        bcy = (y1 + y2) / 2.0
        cx  = frame_w / 2.0
        cy  = frame_h / 2.0

        err_x = bcx - cx
        err_y = bcy - cy

        # PD control
        vx = (err_y * KP - (err_y - self._prev_err_y) * KD)
        vy = (err_x * KP - (err_x - self._prev_err_x) * KD)

        # Clamp speed
        vx = max(-CENTRE_SPEED_MAX, min(CENTRE_SPEED_MAX, vx))
        vy = max(-CENTRE_SPEED_MAX, min(CENTRE_SPEED_MAX, vy))

        # Deadband
        if abs(err_x) < self.centre_thresh // 2: vy = 0.0
        if abs(err_y) < self.centre_thresh // 2: vx = 0.0

        self._prev_err_x = err_x
        self._prev_err_y = err_y
        self._last_update = time.time()

        centred = (abs(err_x) < self.centre_thresh and
                   abs(err_y) < self.centre_thresh)

        if centred:
            self._hover()
            if self._centred_since is None:
                self._centred_since = time.time()
                log.info(f"[Centerer] CENTRED  err=({err_x:+.1f},{err_y:+.1f})px  "
                         f"holding {self.confirm_secs}s...")
                with self._lock:
                    self.state = CenterState.CENTRED

            held = time.time() - self._centred_since
            if held >= self.confirm_secs:
                self._confirm()
        else:
            self._centred_since = None
            self._send_vel(vx, vy, 0.0)
            with self._lock:
                self.state = CenterState.CENTERING

        return self.state

    def stop(self):
        """Abort centering."""
        with self._lock:
            self._running = False
            self.state    = CenterState.IDLE
        self._hover()
        log.info("[Centerer] stopped")

    def _confirm(self):
        """Target confirmed — save snapshot and write acquisition signal."""
        with self._lock:
            self.confirmed = True
            self.state     = CenterState.CONFIRMED
            self._running  = False

        ts = int(time.time())
        seed_out = ACQUIRED_SEEDS_DIR / f"{self.seed_name}_{ts}.jpg"

        if self.snapshot is not None:
            cv2.imwrite(str(seed_out), self.snapshot)
            log.info(f"[Centerer] snapshot saved → {seed_out}")

        # Write acquisition signal for PC to read
        signal = {
            'confirmed'   : True,
            'seed_name'   : self.seed_name,
            'timestamp'   : ts,
            'snapshot_path': str(seed_out),
            'acquired_targets_dir': str(ACQUIRED_TARGETS_DIR),
        }
        ACQUIRE_SIGNAL_FILE.write_text(json.dumps(signal, indent=2))
        log.info(f"[Centerer] ★ TARGET CONFIRMED ★  seed={self.seed_name}")
        log.info(f"[Centerer] Signal → {ACQUIRE_SIGNAL_FILE}")

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                'state'    : self.state,
                'confirmed': self.confirmed,
                'seed_name': self.seed_name,
                'running'  : self._running,
            }


# ─────────────────────────────────────────────────────────────────────────────
# PC-side acquisition handler
# Add this to orb_offload_client.py's frame loop
# ─────────────────────────────────────────────────────────────────────────────

class AcquisitionHandler:
    """
    Runs on the PC. Watches /tmp/target_acquired.json for signals from Pi.
    When signal found → saves current HD frame to acquired_targets/ on PC.
    """

    def __init__(self, save_dir: str = 'acquired_targets'):
        self._save_dir  = Path(save_dir)
        self._save_dir.mkdir(exist_ok=True)
        self._last_ts   = 0
        self._lock      = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running   = True
        self._thread    = threading.Thread(target=self._watch,
                                            name='acq-handler', daemon=True)
        self._thread.start()
        log.info(f"[AcqHandler] watching for acquisition signals → {save_dir}/")

    def push_frame(self, frame_bgr: np.ndarray):
        """Call every frame with the latest decoded frame from Pi."""
        with self._lock:
            self._latest_frame = frame_bgr.copy()

    def stop(self): self._running = False

    def _watch(self):
        sig = Path('/tmp/target_acquired.json')
        while self._running:
            try:
                if sig.exists():
                    data = json.loads(sig.read_text())
                    ts   = data.get('timestamp', 0)
                    if ts != self._last_ts and data.get('confirmed'):
                        self._last_ts = ts
                        self._save_frame(data)
            except Exception as e:
                log.debug(f"[AcqHandler] watch error: {e}")
            time.sleep(0.2)

    def _save_frame(self, signal: dict):
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            log.warning("[AcqHandler] no frame available to save")
            return
        seed_name = signal.get('seed_name', 'target')
        ts        = signal.get('timestamp', int(time.time()))
        out_path  = self._save_dir / f"{seed_name}_{ts}_acquired.jpg"
        cv2.imwrite(str(out_path), frame)
        log.info(f"[AcqHandler] ★ ACQUIRED IMAGE SAVED → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description="Centerer test — simulated lock box")
    p.add_argument('--seed', default='test_seed')
    args = p.parse_args()

    sent_cmds = []

    def _fake_vel(vx, vy, vz):
        sent_cmds.append((vx, vy, vz))
        print(f"  VEL vx={vx:+.3f} vy={vy:+.3f}")

    def _fake_hover():
        print("  HOVER")

    c = Centerer(_fake_vel, _fake_hover, seed_name=args.seed)

    # Fake snapshot
    snap = np.zeros((64, 64, 3), dtype=np.uint8)
    c.start(snap, frame_w=640, frame_h=480)

    print("Simulating centering — lock box moving toward centre...")
    # Simulate lock box moving toward centre over 30 frames
    bx, by = 400.0, 350.0   # start offset from centre
    for i in range(60):
        bx += (320 - bx) * 0.15
        by += (240 - by) * 0.15
        box = (int(bx-30), int(by-30), int(bx+30), int(by+30))
        state = c.update(box, 640, 480)
        print(f"  frame {i:02d}  box_centre=({bx:.0f},{by:.0f})  state={state}")
        if state == CenterState.CONFIRMED:
            print("\n★ CONFIRMED ★")
            break
        time.sleep(0.1)
