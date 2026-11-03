"""
fc_interface.py
═══════════════
Flight controller command interface — Pixhawk via MAVLink.

THIS FILE IS THE ONLY FILE THAT TALKS TO THE PIXHAWK.
Every other module imports from here.

Implementations taken from dronerepl.py (tested on drone).
"""

import logging
import math
import threading
import time

from pymavlink import mavutil

log = logging.getLogger("fc_interface")

# ── Connection state ──────────────────────────────────────────────────────────
_master     = None
_lock       = threading.Lock()

# ── Telemetry ─────────────────────────────────────────────────────────────────
_alt        = 0.0
_alt_source = '?'
_flow_qual  = 0
_flow_vx    = 0.0
_flow_vy    = 0.0
_ekf_flags  = 0
_yaw        = 0.0    # radians, from ATTITUDE
_armed      = False
_mode_str   = 'UNKNOWN'

MODES = {
    'guided':    4,
    'loiter':    5,
    'land':      9,
    'rtl':       6,
    'stabilize': 0,
    'althold':   2,
    'flowhold':  18,
}
COPTER_MODES = {v: k.upper() for k, v in MODES.items()}

# ── Hold state ─────────────────────────────────────────────────────────────────
_hold_active     = False
_hold_thread     = None
_hold_stop_evt   = threading.Event()
_hold_continuous = False
_hold_cont_spd   = 0.0
_hold_cont_vec   = (0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _body_to_ned_vel(dx, dy, speed):
    """Convert body-frame unit vector to NED velocity using live yaw."""
    with _lock:
        yaw = _yaw
    cy = math.cos(yaw); sy = math.sin(yaw)
    vn = cy * dx - sy * dy
    ve = sy * dx + cy * dy
    return vn * speed, ve * speed


def _telemetry_loop():
    global _alt, _alt_source, _flow_qual, _flow_vx, _flow_vy
    global _ekf_flags, _armed, _mode_str, _yaw
    while True:
        try:
            msg = _master.recv_match(
                type=['RANGEFINDER','GLOBAL_POSITION_INT','OPTICAL_FLOW',
                      'EKF_STATUS_REPORT','HEARTBEAT','ATTITUDE'],
                blocking=True, timeout=1)
            if not msg:
                continue
            t = msg.get_type()
            if t == 'RANGEFINDER':
                with _lock:
                    _alt = msg.distance; _alt_source = 'RNGFND'
            elif t == 'GLOBAL_POSITION_INT':
                with _lock:
                    if _alt_source != 'RNGFND':
                        _alt = msg.relative_alt / 1000.0; _alt_source = 'BARO'
            elif t == 'OPTICAL_FLOW':
                with _lock:
                    _flow_qual = msg.quality
                    _flow_vx   = msg.flow_comp_m_x
                    _flow_vy   = msg.flow_comp_m_y
            elif t == 'EKF_STATUS_REPORT':
                with _lock: _ekf_flags = msg.flags
            elif t == 'ATTITUDE':
                with _lock: _yaw = msg.yaw
            elif t == 'HEARTBEAT':
                if msg.get_srcSystem() == _master.target_system:
                    with _lock:
                        _armed    = bool(msg.base_mode &
                                        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        _mode_str = COPTER_MODES.get(msg.custom_mode,
                                                      f'MODE{msg.custom_mode}')
        except Exception as e:
            log.warning(f"Telemetry error: {e}")
            time.sleep(0.5)


def _hold_loop():
    while not _hold_stop_evt.is_set():
        if not _master:
            time.sleep(0.1); continue
        with _lock:
            is_cont = _hold_continuous
            vec     = _hold_cont_vec
            spd     = _hold_cont_spd

        if is_cont:
            dx_unit = vec[0] / (spd or 1.0)
            dy_unit = vec[1] / (spd or 1.0)
            vn, ve  = _body_to_ned_vel(dx_unit, dy_unit, spd)
            vd      = vec[2]
            _master.mav.set_position_target_local_ned_send(
                0, _master.target_system, _master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0, vn, ve, vd,
                0, 0, 0, 0, 0)
            _hold_stop_evt.wait(timeout=0.1)
        else:
            # Soft zero-vel hold
            _master.mav.set_position_target_local_ned_send(
                0, _master.target_system, _master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                0b0000111111000111,
                0, 0, 0, 0.0, 0.0, 0.0,
                0, 0, 0, 0, 0)
            _hold_stop_evt.wait(timeout=0.2)


def _stop_hold():
    global _hold_active, _hold_thread, _hold_continuous, _hold_cont_spd, _hold_cont_vec
    _hold_stop_evt.set()
    if _hold_thread and _hold_thread.is_alive():
        _hold_thread.join(timeout=0.5)
    _hold_active     = False
    _hold_continuous = False
    _hold_cont_spd   = 0.0
    _hold_cont_vec   = (0.0, 0.0, 0.0)


def _start_soft_hold():
    global _hold_active, _hold_thread, _hold_continuous
    _stop_hold()
    _hold_stop_evt.clear()
    _hold_continuous = False
    _hold_active     = True
    _hold_thread = threading.Thread(target=_hold_loop,
                                    name='fc-hold', daemon=True)
    _hold_thread.start()


def _set_mode(mode_name: str):
    mode_id = MODES.get(mode_name.lower())
    if mode_id is None:
        log.warning(f"Unknown mode '{mode_name}'"); return
    _master.mav.command_long_send(
        _master.target_system, _master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id, 0, 0, 0, 0, 0)
    log.info(f"Mode → {mode_name.upper()}")


# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

def connect(port: str = '/dev/ttyUSB0', baud: int = 57600) -> bool:
    global _master
    try:
        log.info(f"Connecting → {port} @ {baud}")
        _master = mavutil.mavlink_connection(port, baud=baud)
        _master.wait_heartbeat()
        log.info(f"Connected — system {_master.target_system}")
        threading.Thread(target=_telemetry_loop,
                         name='fc-telemetry', daemon=True).start()
        return True
    except Exception as e:
        log.error(f"FC connect failed: {e}")
        return False


def disconnect():
    global _master
    _stop_hold()
    try:
        if _master: _master.close()
    except Exception: pass
    _master = None
    log.info("FC disconnected")


def is_connected() -> bool:
    return _master is not None


# ─────────────────────────────────────────────────────────────────────────────
# Arming and mode
# ─────────────────────────────────────────────────────────────────────────────

def arm() -> bool:
    _set_mode('guided')
    time.sleep(0.5)
    _master.mav.command_long_send(
        _master.target_system, _master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    deadline = time.time() + 10
    while time.time() < deadline:
        with _lock:
            if _armed: break
        time.sleep(0.05)
    log.info("ARMED")
    return _armed


def disarm():
    _master.mav.command_long_send(
        _master.target_system, _master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0)
    log.info("DISARMED")


def set_mode(mode: str):
    _set_mode(mode)


# ─────────────────────────────────────────────────────────────────────────────
# Takeoff and landing
# ─────────────────────────────────────────────────────────────────────────────

def takeoff(altitude_m: float = 2.0):
    _set_mode('guided')
    time.sleep(0.3)
    _master.mav.command_long_send(
        _master.target_system, _master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude_m)
    log.info(f"TAKEOFF → {altitude_m}m")
    deadline = time.time() + 30
    while time.time() < deadline:
        with _lock:
            alt = _alt
        if abs(alt - altitude_m) < 0.1:
            break
        time.sleep(0.1)
    log.info(f"TAKEOFF reached {_alt:.2f}m")
    _start_soft_hold()


def land():
    _stop_hold()
    _master.mav.command_long_send(
        _master.target_system, _master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    log.info("LAND")


def rtl():
    _stop_hold()
    _set_mode('rtl')
    log.info("RTL")


# ─────────────────────────────────────────────────────────────────────────────
# Core movement
# ─────────────────────────────────────────────────────────────────────────────

def send_velocity(vx: float, vy: float, vz: float,
                  duration_s: float = 0.1):
    """
    Body-frame velocity → converted to NED via live yaw → sent to FC.
    vx = forward(+) / back(-)
    vy = right(+)   / left(-)
    vz = down(+)    / up(-)   [NED convention]
    """
    _stop_hold()
    spd = math.hypot(vx, vy)
    if spd > 1e-6:
        vn, ve = _body_to_ned_vel(vx / spd, vy / spd, spd)
    else:
        vn, ve = 0.0, 0.0
    _master.mav.set_position_target_local_ned_send(
        0, _master.target_system, _master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0, vn, ve, float(vz),
        0, 0, 0, 0, 0)


def hover():
    """Soft zero-velocity hold — always use this, never tight hold."""
    _start_soft_hold()


def forward(speed_ms: float = 0.2):
    send_velocity(speed_ms, 0.0, 0.0)

def backward(speed_ms: float = 0.2):
    send_velocity(-speed_ms, 0.0, 0.0)

def right(speed_ms: float = 0.2):
    send_velocity(0.0, speed_ms, 0.0)

def left(speed_ms: float = 0.2):
    send_velocity(0.0, -speed_ms, 0.0)

def up(speed_ms: float = 0.2):
    send_velocity(0.0, 0.0, -speed_ms)   # NED: negative vz = up

def down(speed_ms: float = 0.2):
    send_velocity(0.0, 0.0, speed_ms)    # NED: positive vz = down


def yaw(rate_deg_s: float = 20.0, clockwise: bool = True):
    """Rotate in place at rate_deg_s degrees/sec."""
    direction = 1 if clockwise else -1
    _master.mav.command_long_send(
        _master.target_system, _master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        rate_deg_s,   # target angle (used as rate in relative mode)
        rate_deg_s,   # angular speed deg/s
        direction,    # -1=CCW  1=CW
        1,            # 1=relative
        0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry reads
# ─────────────────────────────────────────────────────────────────────────────

def get_altitude() -> float:
    with _lock: return _alt

def get_battery() -> dict:
    # Battery info comes from BATTERY_STATUS or SYS_STATUS
    # returning safe defaults — nav team can extend if needed
    return {'voltage': 0.0, 'current': 0.0, 'level': 100}

def get_heading() -> float:
    with _lock: return math.degrees(_yaw) % 360

def is_armed() -> bool:
    with _lock: return _armed


# ─────────────────────────────────────────────────────────────────────────────
# Safety
# ─────────────────────────────────────────────────────────────────────────────

def emergency_stop():
    log.error("EMERGENCY STOP")
    hover()
    time.sleep(0.3)
    land()


def set_failsafe(action: str = 'RTL'):
    log.info(f"set_failsafe({action}) — configure in FC params directly")


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="fc_interface CLI test")
    parser.add_argument('--port',  default='/dev/ttyUSB0')
    parser.add_argument('--baud',  type=int, default=57600)
    parser.add_argument('--cmd',
                        choices=['hover','forward','backward','left','right',
                                 'up','down','land','rtl','arm','disarm'],
                        default='hover')
    parser.add_argument('--speed', type=float, default=0.2)
    args = parser.parse_args()

    connect(args.port, args.baud)
    {
        'hover'   : hover,
        'forward' : lambda: forward(args.speed),
        'backward': lambda: backward(args.speed),
        'left'    : lambda: left(args.speed),
        'right'   : lambda: right(args.speed),
        'up'      : lambda: up(args.speed),
        'down'    : lambda: down(args.speed),
        'land'    : land,
        'rtl'     : rtl,
        'arm'     : arm,
        'disarm'  : disarm,
    }[args.cmd]()
    time.sleep(1)
    disconnect()
