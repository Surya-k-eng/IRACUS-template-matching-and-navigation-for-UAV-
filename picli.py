
import time
import threading
import json
import datetime
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import socket
from pymavlink import mavutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONNECTION  = '/dev/ttyACM0'
BAUD_RATE   = 57600
STATUS_PORT = 5760

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

master = None

# ── Shared telemetry ──────────────────────────────────────────────────────────
_lock       = threading.Lock()
_alt        = 0.0
_alt_source = '?'
_flow_qual  = 0
_flow_vx    = 0.0
_flow_vy    = 0.0
_ekf_flags  = 0
_yaw        = 0.0          # radians, from ATTITUDE
_armed      = False
_mode_id    = 0
_mode_str   = 'UNKNOWN'
_home_heading = 0.0        # FIX-YAW: set in _cmd_takeoff_blocking, NOT cmd_arm


# ── Logger ────────────────────────────────────────────────────────────────────
_log_lines = []

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    with _lock:
        alt  = _alt
        fq   = _flow_qual
        mode = _mode_str
    line = f"[{ts}] [{mode:>8}] alt={alt:.2f}m flow={fq:>3}  {msg}"
    _log_lines.append(line)
    print(line)

# ── Params ────────────────────────────────────────────────────────────────────
_params        = {}
_params_loaded = False
_params_lock   = threading.Lock()

PARAM_GROUPS = {
    'Flow Hold':  ['FHLD_XY_P','FHLD_XY_I','FHLD_BRAKE_RATE','FHLD_QUAL_MIN','FHLD_FLOW_MAX'],
    'EKF / Flow': ['EK3_SRC1_VELXY','EK3_FLOW_USE','EK3_FLOW_M_NSE','EK3_VELNE_M_NSE','EK3_SRC1_POSZ'],
    'Rangefinder':['RNGFND1_TYPE','RNGFND1_MAX_CM','RNGFND1_MIN_CM','EK3_RNG_M_NSE'],
    'Battery':    ['BATT_LOW_VOLT','BATT_CRT_VOLT','BATT_CAPACITY','BATT_ARM_VOLT'],
    'Rates':      ['ATC_RAT_RLL_P','ATC_RAT_PIT_P','ATC_RAT_YAW_P','PSC_VELXY_P','PSC_POSXY_P'],
}

def fetch_params_background():
    global _params, _params_loaded
    if not master:
        return
    deadline = time.time() + 30
    while time.time() < deadline:
        with _lock:
            if not _armed:
                break
        time.sleep(1.0)
    else:
        print("\n[!] Param fetch skipped — drone stayed armed during 30s window")
        return

    try:
        master.mav.param_request_list_send(master.target_system, master.target_component)
        while True:
            msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)
            if not msg:
                break
            name = msg.param_id.rstrip('\x00')
            with _params_lock:
                _params[name] = float(msg.param_value)
            if len(_params) >= msg.param_count:
                break
        with _params_lock:
            _params_loaded = True
        print(f"\n[+] Params loaded: {len(_params)} params from FC")
    except Exception as e:
        print(f"\n[!] Param fetch error: {e}")


def _ekf_label(flags):
    bits = {0:'ATT', 1:'VELH', 2:'VELV', 3:'POSREL', 5:'POSZ'}
    return '  '.join(f"{'✓' if flags & (1<<b) else '✗'}{n}" for b, n in bits.items())

def _flow_health():
    q = _flow_qual
    if q >= 100: return 'GOOD'
    if q >= 50:  return 'OK'
    if q > 0:    return 'WEAK'
    return 'NONE'

# ── Telemetry thread ──────────────────────────────────────────────────────────
def telemetry_loop():
    global _alt, _alt_source, _flow_qual, _flow_vx, _flow_vy
    global _ekf_flags, _armed, _mode_id, _mode_str, _yaw
    while True:
        try:
            msg = master.recv_match(
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
                with _lock:
                    _ekf_flags = msg.flags
            elif t == 'ATTITUDE':
                with _lock:
                    _yaw = msg.yaw
            elif t == 'HEARTBEAT':
                if msg.get_srcSystem() == master.target_system:
                    with _lock:
                        _armed    = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        _mode_id  = msg.custom_mode
                        _mode_str = COPTER_MODES.get(msg.custom_mode, f'MODE{msg.custom_mode}')
        except Exception as e:
            print(f"\n[!] Telemetry error: {e}")
            time.sleep(0.5)

# ── HTTP dashboard ─────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Drone Status</title>
<style>
  * { box-sizing:border-box; }
  body { background:#0d0d0d; color:#e0e0e0; font-family:monospace; margin:0; padding:1.5rem; }
  h2 { color:#7ef; margin:0 0 1rem; }
  .tabs { display:flex; gap:.5rem; margin-bottom:1.5rem; }
  .tab { padding:.4rem 1rem; border:1px solid #444; border-radius:6px; cursor:pointer; background:#1a1a1a; color:#aaa; }
  .tab.active { background:#7ef; color:#000; border-color:#7ef; }
  .page { display:none; }
  .page.active { display:block; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; max-width:640px; }
  .card { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:1rem; }
  .label { font-size:.75rem; color:#888; text-transform:uppercase; margin-bottom:.25rem; }
  .val { font-size:1.4rem; font-weight:bold; }
  .ARMED { color:#4f4; } .DISARMED { color:#f44; }
  .GOOD { color:#4f4; } .OK { color:#ff4; } .WEAK { color:#f84; } .NONE { color:#f44; }
  .TIGHT { color:#4ff; } .SOFT { color:#fa4; } .OFF { color:#666; } .CONT { color:#f4f; }
  .ekf { font-size:.9rem; line-height:1.8; }
  .ok { color:#4f4; } .bad { color:#f44; }
  #dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         background:#4f4; margin-right:.5rem; animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  #psearch { width:100%; max-width:500px; padding:.5rem; background:#1a1a1a;
             border:1px solid #444; border-radius:6px; color:#e0e0e0;
             font-family:monospace; margin-bottom:1rem; }
  .pgroup { margin-bottom:1.5rem; }
  .pgroup h3 { color:#7ef; font-size:.85rem; margin:0 0 .5rem; text-transform:uppercase; }
  .prow { display:flex; align-items:center; gap:.75rem; padding:.3rem 0; border-bottom:1px solid #1e1e1e; }
  .pname { width:220px; color:#aaa; font-size:.85rem; }
  .pval { width:100px; background:#111; border:1px solid #333; border-radius:4px;
          color:#fff; font-family:monospace; padding:.2rem .4rem; text-align:right; }
  .pval:focus { border-color:#7ef; outline:none; }
  .pset { padding:.2rem .6rem; background:#333; border:1px solid #555;
          border-radius:4px; color:#e0e0e0; cursor:pointer; font-family:monospace; }
  .pset:hover { background:#7ef; color:#000; }
  .pok { color:#4f4; font-size:.8rem; }
  #pstatus { margin-top:.5rem; font-size:.8rem; color:#888; }
  #pallsec { margin-top:2rem; }
  #pallsec h3 { color:#666; font-size:.8rem; margin-bottom:.5rem; }
  .pallrow { display:flex; gap:.75rem; padding:.15rem 0; border-bottom:1px solid #181818; }
  .pallname { width:260px; color:#888; font-size:.8rem; }
  .pallval { color:#ccc; font-size:.8rem; }
</style>
</head>
<body>
<h2><span id="dot"></span>Drone</h2>
<div class="tabs">
  <div class="tab active" onclick="switchTab('telemetry')">Telemetry</div>
  <div class="tab" onclick="switchTab('params')">Params</div>
</div>

<div id="tab-telemetry" class="page active">
<div class="grid">
  <div class="card"><div class="label">Mode</div><div class="val" id="mode">—</div></div>
  <div class="card"><div class="label">Armed</div><div class="val" id="armed">—</div></div>
  <div class="card"><div class="label">Altitude</div><div class="val" id="alt">—</div></div>
  <div class="card">
    <div class="label">Flow Quality</div>
    <div class="val" id="flow_qual">—</div>
    <div id="flow_vel" style="font-size:.8rem;color:#888;margin-top:.25rem"></div>
  </div>
  <div class="card" style="grid-column:span 2">
    <div class="label">EKF Health</div>
    <div class="ekf" id="ekf">—</div>
  </div>
  <div class="card" style="grid-column:span 2">
    <div class="label">Hold</div>
    <div class="val" id="hold">—</div>
  </div>
</div>
</div>

<div id="tab-params" class="page">
  <input id="psearch" type="text" placeholder="Search params..." oninput="filterParams()">
  <div id="pstatus">Loading params from FC...</div>
  <div id="pgroups"></div>
  <div id="pallsec"></div>
</div>

<script>
const EKF_BITS = [[0,'ATT'],[1,'VEL_H'],[2,'VEL_V'],[3,'POS_REL'],[5,'POS_Z']];
let allParams = {};

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['telemetry','params'][i]===name));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  if (name==='params' && Object.keys(allParams).length===0) loadParams();
}

const src = new EventSource('/stream');
src.onmessage = e => {
  try {
    const d = JSON.parse(e.data);
    document.getElementById('mode').textContent = d.mode;
    const aEl = document.getElementById('armed');
    aEl.textContent = d.armed ? 'ARMED' : 'DISARMED';
    aEl.className = 'val ' + (d.armed ? 'ARMED' : 'DISARMED');
    document.getElementById('alt').textContent = d.alt_m.toFixed(3) + ' m  [' + d.alt_source + ']';
    const fEl = document.getElementById('flow_qual');
    fEl.textContent = d.flow_qual + ' / 255  [' + d.flow_health + ']';
    fEl.className = 'val ' + d.flow_health;
    document.getElementById('flow_vel').textContent = 'vx='+d.flow_vx.toFixed(3)+'  vy='+d.flow_vy.toFixed(3);
    const flags = d.ekf_flags;
    document.getElementById('ekf').innerHTML = EKF_BITS.map(([bit,name]) => {
      const ok=(flags>>bit)&1;
      return '<span class="'+(ok?'ok':'bad')+'">'+(ok?'✓':'✗')+' '+name+'</span>';
    }).join('&nbsp;&nbsp;');
    const hEl = document.getElementById('hold');
    hEl.textContent = d.hold_label;
    hEl.className = 'val ' + d.hold_class;
  } catch(ex) { console.error('SSE parse error', ex); }
};
src.onerror = () => { document.getElementById('dot').style.background='#f44'; };

async function loadParams() {
  document.getElementById('pstatus').textContent = 'Fetching from FC...';
  try {
    const r = await fetch('/params');
    const data = await r.json();
    allParams = data.params;
    renderParams(data.groups);
    document.getElementById('pstatus').textContent =
      Object.keys(allParams).length + ' params loaded  •  edit a value and click Set';
  } catch(e) { document.getElementById('pstatus').textContent = 'Error: ' + e; }
}

function renderParams(groups) {
  const gc = document.getElementById('pgroups');
  gc.innerHTML = '';
  for (const [grpName, names] of Object.entries(groups)) {
    const div = document.createElement('div');
    div.className = 'pgroup';
    div.innerHTML = '<h3>' + grpName + '</h3>';
    names.forEach(n => { if (n in allParams) div.appendChild(makeRow(n, allParams[n])); });
    gc.appendChild(div);
  }
  const ac = document.getElementById('pallsec');
  ac.innerHTML = '<h3>All params</h3>';
  Object.entries(allParams).sort((a,b)=>a[0].localeCompare(b[0])).forEach(([n,v]) => {
    const row = document.createElement('div');
    row.className = 'pallrow'; row.dataset.name = n;
    row.innerHTML = '<span class="pallname">'+n+'</span><span class="pallval">'+v+'</span>';
    ac.appendChild(row);
  });
}

function makeRow(name, val) {
  const row = document.createElement('div');
  row.className = 'prow';
  row.innerHTML = '<span class="pname">'+name+'</span>' +
    '<input class="pval" id="pv_'+name+'" value="'+val+'">' +
    '<button class="pset" onclick="setParam(this.dataset.n)" data-n="'+name+'">Set</button>' +
    '<span class="pok" id="ps_'+name+'"></span>';
  return row;
}

async function setParam(name) {
  const val = parseFloat(document.getElementById('pv_'+name).value);
  const s = document.getElementById('ps_'+name);
  s.textContent = '...';
  try {
    const r = await fetch('/setparam', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, value: val})
    });
    const d = await r.json();
    s.textContent = d.ok ? '✓ '+d.value : '✗ '+d.error;
    if (d.ok) allParams[name] = d.value;
  } catch(e) { s.textContent = '✗ '+e; }
}

function filterParams() {
  const q = document.getElementById('psearch').value.toLowerCase();
  document.querySelectorAll('.pallrow').forEach(r => {
    r.style.display = r.dataset.name.toLowerCase().includes(q) ? '' : 'none';
  });
  document.querySelectorAll('.prow').forEach(r => {
    const name = r.querySelector('.pname').textContent;
    r.style.display = name.toLowerCase().includes(q) ? '' : 'none';
  });
}
</script>
</body>
</html>"""


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def get_request(self):
        req, addr = self.socket.accept()
        req.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return req, addr


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    with _lock:
                        h_active = _hold_active
                        h_z      = _hold_z
                        h_cont   = _hold_continuous
                        h_dir    = _hold_cont_dir
                        h_spd    = _hold_cont_spd
                        payload = {
                            'armed':       _armed,
                            'mode':        _mode_str,
                            'alt_m':       round(_alt, 3),
                            'alt_source':  _alt_source,
                            'flow_qual':   _flow_qual,
                            'flow_health': _flow_health(),
                            'flow_vx':     round(_flow_vx, 4),
                            'flow_vy':     round(_flow_vy, 4),
                            'ekf_flags':   _ekf_flags,
                        }
                    if h_cont and h_dir:
                        payload['hold_label'] = f'CONTINUOUS {h_dir.upper()} @ {h_spd:.2f} m/s'
                        payload['hold_class'] = 'CONT'
                    elif h_active and h_z is not None:
                        payload['hold_label'] = f'TIGHT @ {-h_z:.2f} m'
                        payload['hold_class'] = 'TIGHT'
                    elif h_active:
                        payload['hold_label'] = 'SOFT zero-vel'
                        payload['hold_class'] = 'SOFT'
                    else:
                        payload['hold_label'] = 'off'
                        payload['hold_class'] = 'OFF'
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        elif self.path == '/params':
            with _params_lock:
                p = dict(_params)
                loaded = _params_loaded
            body = json.dumps({
                'loaded': loaded, 'count': len(p),
                'params': p, 'groups': PARAM_GROUPS,
            }, indent=2).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/status':
            with _lock:
                payload = {
                    'armed': _armed, 'mode': _mode_str,
                    'alt_m': round(_alt, 3), 'alt_source': _alt_source,
                    'flow_qual': _flow_qual, 'flow_health': _flow_health(),
                    'flow_vx': round(_flow_vx, 4), 'flow_vy': round(_flow_vy, 4),
                    'ekf_flags': _ekf_flags, 'ekf_detail': _ekf_label(_ekf_flags),
                }
            body = json.dumps(payload, indent=2).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == '/setparam':
            length = int(self.headers.get('Content-Length', 0))
            body   = json.loads(self.rfile.read(length))
            name   = body.get('name', '').strip().upper()
            value  = float(body.get('value', 0))
            try:
                master.mav.param_set_send(
                    master.target_system, master.target_component,
                    name.encode('utf-8'), value,
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                ack = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=3)
                if ack and ack.param_id.rstrip('\x00') == name:
                    with _params_lock:
                        _params[name] = float(ack.param_value)
                    resp = json.dumps({'ok': True, 'value': ack.param_value}).encode()
                else:
                    resp = json.dumps({'ok': False, 'error': 'no ack'}).encode()
            except Exception as e:
                resp = json.dumps({'ok': False, 'error': str(e)}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(resp))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, fmt, *args):
        pass


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return '127.0.0.1'


def start_json_backend():
    server = _ThreadedHTTPServer(('0.0.0.0', STATUS_PORT), StatusHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[*] Live dashboard → http://{_get_local_ip()}:{STATUS_PORT}/")


def connect():
    global master
    print(f"[*] Connecting to {CONNECTION} @ {BAUD_RATE}...")
    master = mavutil.mavlink_connection(CONNECTION, baud=BAUD_RATE)
    master.wait_heartbeat()
    print(f"[+] Connected — system {master.target_system}")
    threading.Thread(target=telemetry_loop, daemon=True).start()
    print("[*] Telemetry: RANGEFINDER + OPTICAL_FLOW + EKF_STATUS_REPORT")
    start_json_backend()
    threading.Timer(5.0, fetch_params_background).start()
    print("[*] Fetching params from FC in background (5s delay, waits for disarmed)...")


def set_mode(mode_name):
    mode_id = MODES.get(mode_name.lower())
    if mode_id is None:
        print(f"[!] Unknown mode '{mode_name}'. Options: {list(MODES.keys())}")
        return
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id, 0, 0, 0, 0, 0)
    print(f"[+] Mode → {mode_name.upper()}")


# ── Override / action system ───────────────────────────────────────────────────
_action_thread = None
_action_cancel = threading.Event()

def _cancel_action(wait=True):
    global _action_thread
    _action_cancel.set()
    if wait and _action_thread and _action_thread.is_alive():
        _action_thread.join(timeout=1.0)

def _run_action(fn, *args):
    global _action_thread
    _cancel_action(wait=True)
    _action_cancel.clear()
    _action_thread = threading.Thread(target=fn, args=args, daemon=True)
    _action_thread.start()


# ── Hold system ────────────────────────────────────────────────────────────────
_hold_active     = False
_hold_thread     = None
_hold_z          = None
_hold_continuous = False
_hold_cont_dir   = None
_hold_cont_spd   = 0.0
_hold_cont_vec   = (0.0, 0.0, 0.0)   # body-relative unit vector × speed
_hold_stop_evt   = threading.Event()


# FIX-CONT: convert body-relative direction vector to NED using current yaw,
# then send as LOCAL_NED velocity-only.  This prevents ArduPilot treating
# BODY_OFFSET_NED as a position command when the drone is close to a wall
# or the position controller hits a limit.
def _body_to_ned_vel(dx, dy, speed):
    """
    dx, dy: unit vector in body frame (fwd=+x, right=+y)
    Returns (vn, ve) in m/s in NED frame.
    """
    with _lock:
        yaw = _yaw   # radians
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    vn = cy * dx - sy * dy
    ve = sy * dx + cy * dy
    return vn * speed, ve * speed


def _hold_loop():
    while not _hold_stop_evt.is_set():
        if not master:
            time.sleep(0.1)
            continue
        with _lock:
            is_cont = _hold_continuous
            vec     = _hold_cont_vec   # (dx, dy, dz) unit + speed baked in
            z       = _hold_z

        if is_cont:
            # FIX-CONT: re-project body→NED every packet so turns stay correct
            dx_unit = vec[0] / (_hold_cont_spd or 1.0)
            dy_unit = vec[1] / (_hold_cont_spd or 1.0)
            vn, ve  = _body_to_ned_vel(dx_unit, dy_unit, _hold_cont_spd)
            vd      = vec[2]   # up/down already in NED sign convention

            # type_mask: ignore pos + accel, use velocity only
            # 0b0000111111000111 = ignore pos(xyz) + accel(xyz), use vel(xyz)
            master.mav.set_position_target_local_ned_send(
                0, master.target_system, master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # FIX-CONT: NED not BODY_OFFSET
                0b0000111111000111,
                0, 0, 0,          # pos ignored
                vn, ve, vd,       # velocity in NED
                0, 0, 0, 0, 0)
            _hold_stop_evt.wait(timeout=0.1)

        elif z is not None:
            master.mav.set_position_target_local_ned_send(
                0, master.target_system, master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000110111100011,
                0, 0, z, 0.0, 0.0, 0.0,
                0, 0, 0, 0, 0)
            _hold_stop_evt.wait(timeout=0.1)

        else:
            # Soft zero-vel hold in body frame is fine — no position pin
            master.mav.set_position_target_local_ned_send(
                0, master.target_system, master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                0b0000111111000111,
                0, 0, 0, 0.0, 0.0, 0.0,
                0, 0, 0, 0, 0)
            _hold_stop_evt.wait(timeout=0.2)


def _stop_hold():
    global _hold_active, _hold_thread
    global _hold_continuous, _hold_cont_dir, _hold_cont_spd, _hold_cont_vec
    _hold_stop_evt.set()
    if _hold_thread and _hold_thread.is_alive():
        _hold_thread.join(timeout=0.5)
    _hold_active     = False
    _hold_continuous = False
    _hold_cont_dir   = None
    _hold_cont_spd   = 0.0
    _hold_cont_vec   = (0.0, 0.0, 0.0)


def _start_hold(ned_z=None):
    global _hold_active, _hold_thread, _hold_z, _hold_continuous
    _stop_hold()
    _hold_stop_evt.clear()
    _hold_z          = ned_z
    _hold_active     = True
    _hold_continuous = False
    _hold_thread = threading.Thread(target=_hold_loop, daemon=True)
    _hold_thread.start()


def _start_continuous(dx, dy, dz, speed, direction_str):
    """
    dx, dy, dz: unit direction in body frame.
    speed: m/s magnitude.
    _hold_cont_vec stores (dx*speed, dy*speed, dz*speed) so the loop can
    re-project to NED each tick using the live yaw.
    """
    global _hold_active, _hold_thread, _hold_z
    global _hold_continuous, _hold_cont_dir, _hold_cont_spd, _hold_cont_vec
    _stop_hold()
    _hold_stop_evt.clear()
    _hold_z          = None
    _hold_continuous = True
    _hold_cont_dir   = direction_str
    _hold_cont_spd   = speed
    _hold_cont_vec   = (dx * speed, dy * speed, dz * speed)
    _hold_active     = True
    _hold_thread = threading.Thread(target=_hold_loop, daemon=True)
    _hold_thread.start()


def cmd_stop():
    _cancel_action(wait=False)
    _start_hold(ned_z=None)
    log(f"STOP — soft zero-vel hold at {_alt:.2f} m")


# ── Arm / takeoff / land / RTL ────────────────────────────────────────────────
def cmd_arm():
    # FIX-YAW: do NOT snapshot heading here — EKF yaw unstable before motors spin.
    # Heading snapshot happens in _cmd_takeoff_blocking after GUIDED mode is active.
    set_mode('guided')
    time.sleep(0.5)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0)
    deadline = time.time() + 10
    while time.time() < deadline:
        with _lock:
            if _armed:
                break
        time.sleep(0.05)
    log("ARMED")


def _cmd_takeoff_blocking(alt):
    global _home_heading
    target = float(alt)
    set_mode('guided')
    time.sleep(0.3)

    # FIX-YAW: snapshot heading NOW — GUIDED is active, EKF yaw is settled,
    # motors haven't started spinning yet.  Much more reliable than pre-arm.
    with _lock:
        _home_heading = math.degrees(_yaw) % 360
    log(f"HOME HEADING snapshot (pre-climb): {_home_heading:.1f}°")

    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, target)
    log(f"TAKEOFF target={target} m  (override anytime)")
    deadline = time.time() + 30

    while time.time() < deadline and not _action_cancel.is_set():
        print(f"    alt: {_alt:.2f}/{target} m  flow: {_flow_qual}/255    ", end='\r', flush=True)
        if abs(_alt - target) < 0.05:
            print()
            log("TAKEOFF reached — waiting 3s for EKF settle...")
            settle_deadline = time.time() + 3.0
            while time.time() < settle_deadline:
                if _action_cancel.is_set():
                    return
                time.sleep(0.1)

            with _lock:
                stable_heading = math.degrees(_yaw) % 360

            # Shortest-path angular error: positive = drone rotated CW from home
            error = (_home_heading - stable_heading + 360) % 360
            if error > 180:
                error -= 360

            # FIX-YAW: direction decision now logged so you can verify sign
            direction = 'right' if error > 0 else 'left'
            log(f"HEADING — home={_home_heading:.1f}° stable={stable_heading:.1f}° "
                f"error={error:+.1f}°")
            log(f"TURN DECISION: error={error:+.1f}° → turning {direction} {abs(error):.1f}°")

            if abs(error) > 3.0:
                _cmd_turn_blocking(direction, abs(error), 30.0)
            else:
                log("HEADING OK — no correction needed")

            with _lock:
                ned_z = -_alt
            log(f"TIGHT HOLD at {_alt:.2f} m")
            _start_hold(ned_z=ned_z)
            return
        time.sleep(0.05)

    if _action_cancel.is_set():
        print()
        log(f"TAKEOFF OVERRIDDEN at {_alt:.2f} m")
        return
    print(f"\n[!] Takeoff timeout — alt {_alt:.2f} m [{_alt_source}]")


def cmd_takeoff(alt):
    _run_action(_cmd_takeoff_blocking, alt)


def cmd_land():
    _cancel_action(wait=False)
    _stop_hold()
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
        0, 0, 0, 0, 0, 0, 0)
    log("LAND")


def cmd_rtl():
    _cancel_action(wait=False)
    _stop_hold()
    set_mode('rtl')
    print("[*] RTL engaged.")


def cmd_disarm():
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        0, 0, 0, 0, 0, 0, 0)
    print("[+] Disarmed.")


# ── Velocity / move ────────────────────────────────────────────────────────────
def cmd_velocity(vx, vy, vz=0.0):
    """One-shot velocity in body frame (converted to NED for the FC)."""
    _stop_hold()
    vn, ve = _body_to_ned_vel(vx / (math.hypot(vx, vy) or 1.0),
                               vy / (math.hypot(vx, vy) or 1.0),
                               math.hypot(vx, vy))
    master.mav.set_position_target_local_ned_send(
        0, master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,
        0, 0, 0, vn, ve, float(vz),
        0, 0, 0, 0, 0)


MOVE_DIRECTIONS = {
    'fwd':   ( 1,  0,  0),
    'back':  (-1,  0,  0),
    'right': ( 0,  1,  0),
    'left':  ( 0, -1,  0),
    'up':    ( 0,  0, -1),
    'down':  ( 0,  0,  1),
}


def _cmd_move_blocking(parts):
    if len(parts) < 3:
        print("[!] Usage: move <fwd|back|left|right|up|down> <metres|speed>  [c]")
        return
    direction = parts[1]
    if direction not in MOVE_DIRECTIONS:
        print(f"[!] Unknown direction '{direction}'."); return

    dx, dy, dz = MOVE_DIRECTIONS[direction]

    # ── Continuous mode ──────────────────────────────────────────────────────
    # FIX-CONT: _start_continuous now sends LOCAL_NED velocity reprojected each
    # tick.  No distance tracking, no timeout — runs until next command.
    if len(parts) > 3 and parts[3] == 'c':
        speed = float(parts[2])
        _start_continuous(dx, dy, dz, speed, direction)
        log(f"MOVE {direction.upper()} CONTINUOUS @ {speed} m/s — type any command to stop")
        return

    if len(parts) > 3 and parts[3] == 'spd':
        speed = float(parts[2])
        _stop_hold()
        cmd_velocity(dx * speed, dy * speed, dz * speed)
        log(f"MOVE {direction} one-shot @ {speed} m/s")
        return

    # ── Distance mode (flow-tracked) ─────────────────────────────────────────
    with _lock:
        prev_hold_z = _hold_z

    dist     = float(parts[2])
    v_target = 0.15
    Kp       = 0.8
    v_min    = 0.05
    v_max    = 0.30
    timeout  = (dist / v_target) * 3.0

    _stop_hold()
    log(f"MOVE {direction} {dist*100:.0f} cm")
    cmd_velocity(dx * v_target, dy * v_target, dz * v_target)

    travelled  = 0.0
    t_start    = time.time()
    _prev_flow = (None, None)

    while not _action_cancel.is_set():
        time.sleep(0.05)
        elapsed = time.time() - t_start
        with _lock:
            vx = _flow_vx; vy = _flow_vy; fq = _flow_qual

        cur_flow = (vx, vy)
        if cur_flow != _prev_flow:
            v_actual   = vx * dx + vy * dy
            travelled += abs(v_actual * 0.05)
            _prev_flow = cur_flow
            if fq >= 50:
                v_error = v_target - abs(v_actual)
                v_cmd   = max(v_min, min(v_max, v_target + Kp * v_error))
                cmd_velocity(dx * v_cmd, dy * v_cmd, dz * v_cmd)

        if travelled >= dist:
            log(f"MOVE {direction} done via FLOW ({travelled*100:.1f} cm in {elapsed:.2f}s)"); break
        if elapsed >= timeout:
            log(f"MOVE {direction} done via TIMER ({travelled*100:.1f} cm flow-measured)"); break

    if _action_cancel.is_set():
        log(f"MOVE {direction} OVERRIDDEN at {travelled*100:.1f} cm"); return

    _start_hold(ned_z=prev_hold_z)
    log(f"MOVE done — {'TIGHT' if prev_hold_z is not None else 'SOFT'} hold")


def cmd_move(parts):
    _run_action(_cmd_move_blocking, parts)


# ── Turn ───────────────────────────────────────────────────────────────────────
TURN_DIR = {'left': -1, 'right': 1}


def _cmd_turn_blocking(direction, degrees, speed):
    with _lock:
        prev_hold_z = _hold_z
    _stop_hold()

    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        degrees, speed, TURN_DIR[direction], 1,
        0, 0, 0)
    log(f"TURN {direction.upper()} {degrees:.0f}° @ {speed:.0f}°/s")

    with _lock:
        start_yaw = _yaw
    target_delta = math.radians(degrees)
    timeout  = (degrees / speed) * 1.5 + 1.0
    deadline = time.time() + timeout

    while not _action_cancel.is_set() and time.time() < deadline:
        time.sleep(0.05)
        with _lock:
            current_yaw = _yaw
        delta = abs(math.atan2(
            math.sin(current_yaw - start_yaw),
            math.cos(current_yaw - start_yaw)))
        if delta >= target_delta - math.radians(3):
            break

    if _action_cancel.is_set():
        log("TURN OVERRIDDEN"); return

    log(f"TURN done")
    _start_hold(ned_z=prev_hold_z)


def cmd_turn(parts):
    if len(parts) < 2 or parts[1] not in TURN_DIR:
        print("[!] Usage: turn <left|right> [degrees]  (default 90°)")
        return
    direction = parts[1]
    degrees   = float(parts[2]) if len(parts) > 2 else 90.0
    _run_action(_cmd_turn_blocking, direction, degrees, 30.0)


# ── Goto ───────────────────────────────────────────────────────────────────────
def _cmd_goto_blocking(target_alt):
    with _lock:
        current = _alt

    delta   = target_alt - current
    v_sign  = -1.0 if delta > 0 else 1.0
    v_climb = 0.3
    timeout = (abs(delta) / v_climb) * 2.5 + 2.0

    _stop_hold()
    log(f"GOTO alt={target_alt:.2f} m  (currently {current:.2f} m)")

    deadline = time.time() + timeout
    while not _action_cancel.is_set() and time.time() < deadline:
        master.mav.set_position_target_local_ned_send(
            0, master.target_system, master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0, 0.0, 0.0, v_sign * v_climb,
            0, 0, 0, 0, 0)
        time.sleep(0.1)
        with _lock:
            cur = _alt
        print(f"    alt: {cur:.2f}/{target_alt:.2f} m    ", end='\r', flush=True)
        if abs(cur - target_alt) < 0.08:
            print()
            log(f"GOTO reached {cur:.2f} m — SOFT HOLD")
            _start_hold(ned_z=None)
            return

    print()
    if _action_cancel.is_set():
        log(f"GOTO OVERRIDDEN at {_alt:.2f} m"); return
    log(f"GOTO timeout — reached {_alt:.2f} m")
    _start_hold(ned_z=None)


def cmd_goto(parts):
    if len(parts) < 2:
        print("[!] Usage: goto <metres>"); return
    try:
        target = float(parts[1])
    except ValueError:
        print("[!] goto: altitude must be a number"); return
    if target < 0.2:
        print("[!] goto: minimum altitude 0.2 m"); return
    _run_action(_cmd_goto_blocking, target)


# ── Mission runner ────────────────────────────────────────────────────────────
_mission_abort = threading.Event()


def _run_move_timed(direction, speed, duration):
    """Continuous velocity for `duration` seconds then soft hold."""
    if direction not in MOVE_DIRECTIONS:
        log(f"MISSION bad direction '{direction}'"); return
    dx, dy, dz = MOVE_DIRECTIONS[direction]
    _start_continuous(dx, dy, dz, speed, direction)
    deadline = time.time() + duration
    while time.time() < deadline:
        if _mission_abort.is_set() or _action_cancel.is_set():
            return   # caller handles hold
        time.sleep(0.05)
    # FIX-CONT: _start_hold was inside the loop before — now correctly OUTSIDE
    _start_hold(ned_z=None)


def _run_mission_blocking(path):
    global _mission_abort
    _mission_abort.clear()

    try:
        with open(path) as f:
            mission = json.load(f)
    except Exception as e:
        print(f"[!] Mission load error: {e}"); return

    name  = mission.get('name', path)
    steps = mission.get('steps', [])
    log(f"MISSION START — '{name}'  ({len(steps)} steps)")

    for s in steps:
        if _mission_abort.is_set():
            log("MISSION ABORTED"); return

        step = s.get('step', '?')
        cmd  = s.get('cmd', '')
        log(f"MISSION step {step}: {cmd.upper()} {dict((k,v) for k,v in s.items() if k not in ('step','cmd'))}")

        if cmd == 'arm':
            cmd_arm()
            time.sleep(1.0)

        elif cmd == 'takeoff':
            alt = float(s.get('alt', 1.0))
            _action_cancel.clear()
            _cmd_takeoff_blocking(alt)
            time.sleep(0.5)

        elif cmd == 'move':
            direction = s.get('dir', 'fwd')
            speed     = float(s.get('speed', 0.3))
            duration  = float(s.get('duration', 3.0))
            _run_move_timed(direction, speed, duration)
            time.sleep(0.3)

        elif cmd == 'turn':
            direction = s.get('dir', 'right')
            degrees   = float(s.get('deg', 90.0))
            _action_cancel.clear()
            _cmd_turn_blocking(direction, degrees, 30.0)
            time.sleep(0.3)

        elif cmd == 'goto':
            target = float(s.get('alt', 1.0))
            _action_cancel.clear()
            _cmd_goto_blocking(target)
            time.sleep(0.3)

        elif cmd == 'land':
            cmd_land()
            time.sleep(0.5)

        elif cmd == 'rtl':
            cmd_rtl()
            time.sleep(0.5)

        elif cmd == 'wait':
            secs = float(s.get('duration', 1.0))
            log(f"MISSION wait {secs}s")
            deadline = time.time() + secs
            while time.time() < deadline:
                if _mission_abort.is_set(): return
                time.sleep(0.1)

        else:
            log(f"MISSION unknown cmd '{cmd}' — skipping")

    log(f"MISSION COMPLETE — '{name}'")


def cmd_mission(parts):
    if len(parts) < 2:
        print("[!] Usage: mission <path/to/file.json>"); return
    path = parts[1]
    _run_action(_run_mission_blocking, path)


def cmd_mission_abort():
    _mission_abort.set()
    _cancel_action(wait=False)
    _stop_hold()
    log("MISSION ABORTED by user")


# ── Params / tune ──────────────────────────────────────────────────────────────
def _set_param(name, value):
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode('utf-8'), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ack = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)
    if ack and ack.param_id.rstrip('\x00') == name:
        print(f"  {name} = {ack.param_value}")
    else:
        print(f"  [!] No ack for {name}")


def cmd_tune():
    print("[*] Applying flow-hold tuning params...")
    _set_param('FHLD_XY_P', 0.35)
    _set_param('FHLD_BRAKE_RATE', 10.0)
    _set_param('EK3_FLOW_M_NSE', 0.15)
    print("[+] Done.")


# ── Status / monitor ───────────────────────────────────────────────────────────
def cmd_status():
    with _lock:
        armed=_armed; mode=_mode_str; alt=_alt; src=_alt_source
        fq=_flow_qual; fvx=_flow_vx; fvy=_flow_vy; ekf=_ekf_flags
    active = _action_thread and _action_thread.is_alive()
    if _hold_continuous and _hold_cont_dir:
        hm = f"CONTINUOUS {_hold_cont_dir.upper()} @ {_hold_cont_spd:.2f} m/s"
    elif _hold_active and _hold_z is not None:
        hm = f"TIGHT @ NED Z={_hold_z:.2f} m"
    elif _hold_active:
        hm = "SOFT zero-vel"
    else:
        hm = "off"
    print(f"  MODE    : {mode}  ({'ARMED' if armed else 'DISARMED'})")
    print(f"  ALT     : {alt:.2f} m  [{src}]")
    print(f"  FLOW    : qual={fq}/255  [{_flow_health()}]  vx={fvx:.3f}  vy={fvy:.3f}")
    print(f"  EKF     : {_ekf_label(ekf)}  (raw={ekf:#010b})")
    print(f"  ACTION  : {'running' if active else 'idle'}")
    print(f"  HOLD    : {hm}")


def cmd_monitor():
    print("[*] Monitor — Ctrl+C to return\n")
    try:
        while True:
            with _lock:
                armed=_armed; mode=_mode_str; alt=_alt; src=_alt_source
                fq=_flow_qual; fvx=_flow_vx; fvy=_flow_vy; ekf=_ekf_flags
            active = _action_thread and _action_thread.is_alive()
            if _hold_continuous and _hold_cont_dir:
                hm = f"CONT {_hold_cont_dir.upper()}@{_hold_cont_spd:.2f}m/s"
            elif _hold_active and _hold_z is not None:
                hm = f"TIGHT@{-_hold_z:.2f}m"
            elif _hold_active:
                hm = "SOFT"
            else:
                hm = "off"
            lines = [
                f"  ┌─────────────── DRONE TELEMETRY ───────────────┐",
                f"  │  MODE    {mode:<10}  {'[ARMED]  ' if armed else '[DISARMED]'}         │",
                f"  │  ALT     {alt:.3f} m   src={src:<6}               │",
                f"  │  FLOW    qual={fq:>3}/255  [{_flow_health():<4}]  vx={fvx:+.3f} vy={fvy:+.3f}  │",
                f"  │  EKF     {_ekf_label(ekf):<38}  │",
                f"  │  ACTION  {'RUNNING' if active else 'idle':<38}  │",
                f"  │  HOLD    {hm:<38}  │",
                f"  └────────────────────────────────────────────────┘",
            ]
            print('\033[F' * 8 if hasattr(cmd_monitor, '_started') else '', end='')
            cmd_monitor._started = True
            print('\n'.join(lines), flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        del cmd_monitor._started
        print("\n[*] Exited monitor.")


# ── REPL ───────────────────────────────────────────────────────────────────────
HELP = """
Commands:
  arm                          → GUIDED + arm
  takeoff <m>                  → climb to altitude; auto-engages TIGHT hold
  land / x                     → land  [cancels active action]
  stop                         → soft zero-vel hold  [cancels active action]
  turn <left|right> [deg]      → yaw relative (default 90°)
  move <dir> <dist>            → fwd/back/left/right/up/down + metres (flow-tracked)
  move <dir> <spd> c           → CONTINUOUS velocity — keeps going until next command
  move <dir> <spd> spd         → one-shot velocity send
  mission <file.json>          → run autonomous mission file
  missionabort                 → abort running mission immediately
  rtl                          → Return To Launch
  mode <name>                  → guided/loiter/flowhold/althold/stabilize/rtl/land
  disarm                       → force disarm
  tune                         → set FHLD / EK3 tuning params
  status                       → telemetry snapshot
  monitor                      → live terminal dashboard  (Ctrl+C to return)
  help / quit / exit
"""


def main():
    connect()
    print(HELP)
    while True:
        try:
            raw = input("drone> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Exiting."); break
        if not raw:
            continue
        parts = raw.lower().split()
        cmd   = parts[0]

        if cmd in ('quit', 'exit'):   print("[*] Bye."); break
        elif cmd == 'help':           print(HELP)
        elif cmd == 'arm':            cmd_arm()
        elif cmd == 'takeoff':        cmd_takeoff(float(parts[1]) if len(parts) > 1 else 1.0)
        elif cmd in ('land', 'x'):    cmd_land()
        elif cmd == 'stop':           cmd_stop()
        elif cmd in ('move', 'mv'):   cmd_move(parts)
        elif cmd == 'turn':           cmd_turn(parts)
        elif cmd == 'goto':           cmd_goto(parts)
        elif cmd == 'mission':        cmd_mission(parts)
        elif cmd == 'missionabort':   cmd_mission_abort()
        elif cmd == 'rtl':            cmd_rtl()
        elif cmd == 'disarm':         cmd_disarm()
        elif cmd == 'tune':           cmd_tune()
        elif cmd == 'status':         cmd_status()
        elif cmd == 'monitor':        cmd_monitor()
        elif cmd == 'mode':
            if len(parts) > 1: _cancel_action(wait=False); set_mode(parts[1])
            else: print("[!] Usage: mode <name>")
        else:
            _stop_hold()
            print(f"[!] Unknown command '{cmd}'. Type 'help'.")


if __name__ == '__main__':
    main()


