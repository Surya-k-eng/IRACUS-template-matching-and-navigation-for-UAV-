#!/usr/bin/env python3
"""
autopilot.py
════════════
ASCEND — Autopilot + REPL + Unified Web Dashboard

Web dashboard at http://<pi-ip>:5760/
  ┌─────────────────────────────┬──────────┐
  │  MAIN STREAM (large)        │ TELEMETRY│
  │  • Takeoff  → raw picam     │ MODE     │
  │  • Survey   → ORB stream    │ ALT      │
  │  • Landing  → QR stream     │ FLOW     │
  ├─────────────────────────────┤ EKF      │
  │  BOUNDARY (small, always)   │ AP STATE │
  └─────────────────────────────┴──────────┘

REPL commands:
  start                → begin autonomous mission
  abort                → smooth stop + land
  qrland               → trigger QR landing now
  boundary sim         → start boundary detection (sim, no FC)
  boundary real        → start boundary detection (real FC)
  boundary stop        → stop boundary detection
  arm / takeoff <m> / land / x / stop / rtl / disarm
  move <dir> <dist|spd> [c] / turn <left|right> [deg]
  goto <m> / mode <name> / tune / status / monitor
  help / quit
"""

import datetime, json, math, os, socket, struct, threading, time, sys
from enum import Enum, auto
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from queue import Queue, Empty

from pymavlink import mavutil

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONNECTION   = '/dev/ttyACM0'
BAUD_RATE    = 57600
DASHBOARD_PORT = 5760

# Stream ports
PORT_RAW     = 8081   # raw picam (takeoff)
PORT_ORB     = 8082   # orb annotated (survey) — fed by orb_main.py
PORT_QR      = 8083   # qr landing stream
PORT_BOUNDARY= 8084   # boundary detection stream

SURVEY_ALT      = 2.0
SURVEY_SPEED    = 0.4
CONFIRM_SECS    = 4.0
YAW_360_SPEED   = 25.0
RETURN_SPEED    = 0.4
RETURN_SECS     = 8.0
QR_LAND_TIMEOUT = 30.0
QR_CENTRE_THRESH= 30
QR_SEED         = 'seedqr.jpeg'
QR_CAM          = 0

MODES = {
    'guided':4,'loiter':5,'land':9,'rtl':6,
    'stabilize':0,'althold':2,'flowhold':18,
}
COPTER_MODES = {v:k.upper() for k,v in MODES.items()}
MOVE_DIRECTIONS = {
    'fwd':(1,0,0),'back':(-1,0,0),'right':(0,1,0),
    'left':(0,-1,0),'up':(0,0,-1),'down':(0,0,1),
}
TURN_DIR = {'left':-1,'right':1}

# ─────────────────────────────────────────────────────────────────────────────
# Shared telemetry state
# ─────────────────────────────────────────────────────────────────────────────

master      = None
_lock       = threading.Lock()
_alt        = 0.0
_alt_source = '?'
_flow_qual  = 0
_flow_vx    = 0.0
_flow_vy    = 0.0
_ekf_flags  = 0
_yaw        = 0.0
_armed      = False
_mode_str   = 'UNKNOWN'
_home_heading = 0.0
_ap_state_label = 'IDLE'
_active_stream  = 'raw'      # 'raw' | 'orb' | 'qr'
_boundary_status = {}        # latest boundary result dict

_params      = {}
_params_loaded = False
_params_lock = threading.Lock()
PARAM_GROUPS = {
    'Flow Hold':['FHLD_XY_P','FHLD_XY_I','FHLD_BRAKE_RATE','FHLD_QUAL_MIN'],
    'EKF':['EK3_SRC1_VELXY','EK3_FLOW_USE','EK3_FLOW_M_NSE'],
    'Rangefinder':['RNGFND1_TYPE','RNGFND1_MAX_CM','RNGFND1_MIN_CM'],
    'Rates':['ATC_RAT_RLL_P','ATC_RAT_PIT_P','ATC_RAT_YAW_P'],
}

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

_log_lines = []
def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    with _lock: alt=_alt; fq=_flow_qual; mode=_mode_str
    line = f"[{ts}] [{mode:>8}] alt={alt:.2f}m flow={fq:>3}  {msg}"
    _log_lines.append(line)
    print(line)

# ─────────────────────────────────────────────────────────────────────────────
# Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def _ekf_label(f):
    bits={0:'ATT',1:'VELH',2:'VELV',3:'POSREL',5:'POSZ'}
    return '  '.join(f"{'✓' if f&(1<<b) else '✗'}{n}" for b,n in bits.items())

def _flow_health():
    q=_flow_qual
    if q>=100: return 'GOOD'
    if q>=50:  return 'OK'
    if q>0:    return 'WEAK'
    return 'NONE'

def telemetry_loop():
    global _alt,_alt_source,_flow_qual,_flow_vx,_flow_vy,_ekf_flags,_armed,_mode_str,_yaw
    while True:
        try:
            msg=master.recv_match(
                type=['RANGEFINDER','GLOBAL_POSITION_INT','OPTICAL_FLOW',
                      'EKF_STATUS_REPORT','HEARTBEAT','ATTITUDE'],
                blocking=True,timeout=1)
            if not msg: continue
            t=msg.get_type()
            if t=='RANGEFINDER':
                with _lock: _alt=msg.distance; _alt_source='RNGFND'
            elif t=='GLOBAL_POSITION_INT':
                with _lock:
                    if _alt_source!='RNGFND': _alt=msg.relative_alt/1000.0; _alt_source='BARO'
            elif t=='OPTICAL_FLOW':
                with _lock: _flow_qual=msg.quality; _flow_vx=msg.flow_comp_m_x; _flow_vy=msg.flow_comp_m_y
            elif t=='EKF_STATUS_REPORT':
                with _lock: _ekf_flags=msg.flags
            elif t=='ATTITUDE':
                with _lock: _yaw=msg.yaw
            elif t=='HEARTBEAT':
                if msg.get_srcSystem()==master.target_system:
                    with _lock:
                        _armed=bool(msg.base_mode&mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        _mode_str=COPTER_MODES.get(msg.custom_mode,f'MODE{msg.custom_mode}')
        except Exception as e:
            print(f"\n[!] Telemetry: {e}"); time.sleep(0.5)

def fetch_params_background():
    global _params,_params_loaded
    if not master: return
    deadline=time.time()+30
    while time.time()<deadline:
        with _lock:
            if not _armed: break
        time.sleep(1.0)
    else: return
    try:
        master.mav.param_request_list_send(master.target_system,master.target_component)
        while True:
            msg=master.recv_match(type='PARAM_VALUE',blocking=True,timeout=2)
            if not msg: break
            with _params_lock: _params[msg.param_id.rstrip('\x00')]=float(msg.param_value)
            if len(_params)>=msg.param_count: break
        with _params_lock: _params_loaded=True
        print(f"\n[+] Params: {len(_params)} loaded")
    except Exception as e:
        print(f"\n[!] Params: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Raw picam MJPEG streamer (takeoff stream)
# ─────────────────────────────────────────────────────────────────────────────

class RawCamStreamer:
    """Streams raw picam frames on PORT_RAW."""
    def __init__(self):
        self._jpg     = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self._srv     = None

    def start(self):
        import http.server, socketserver as _ss
        streamer = self
        class _H(http.server.BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_GET(self):
                if self.path=='/stream':
                    self.send_response(200)
                    self.send_header('Content-Type','multipart/x-mixed-replace; boundary=--jpgbnd')
                    self.end_headers()
                    try:
                        while streamer._running:
                            with streamer._lock: jpg=streamer._jpg
                            if jpg is None: time.sleep(0.02); continue
                            self.wfile.write(b'--jpgbnd\r\nContent-Type: image/jpeg\r\n'
                                +f'Content-Length: {len(jpg)}\r\n\r\n'.encode()+jpg+b'\r\n')
                            time.sleep(0.033)
                    except (BrokenPipeError,ConnectionResetError): pass
                else: self.send_response(404); self.end_headers()
        class _S(_ss.ThreadingTCPServer):
            allow_reuse_address=True
        self._srv=_S(('0.0.0.0',PORT_RAW),_H)
        self._running=True
        threading.Thread(target=self._srv.serve_forever,daemon=True).start()
        threading.Thread(target=self._capture,daemon=True).start()
        log(f"[RawCam] stream → :{PORT_RAW}/stream")

    def _capture(self):
        import cv2
        try:
            from picamera2 import Picamera2
            cam=Picamera2(0)
            cfg=cam.create_preview_configuration(main={"size":(640,480),"format":"RGB888"})
            cam.configure(cfg); cam.start(); time.sleep(1.0)
            for _ in range(5): cam.capture_array()
            while self._running:
                arr=cam.capture_array()
                if arr is not None:
                    _,jpg=cv2.imencode('.jpg',arr,[cv2.IMWRITE_JPEG_QUALITY,70])
                    with self._lock: self._jpg=jpg.tobytes()
                time.sleep(0.033)
            cam.stop(); cam.close()
        except Exception as e:
            log(f"[RawCam] error: {e}")

    def stop(self):
        self._running=False
        if self._srv: self._srv.shutdown()

# ─────────────────────────────────────────────────────────────────────────────
# Unified Web Dashboard
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ASCEND</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0a;color:#e0e0e0;font-family:monospace;height:100vh;display:flex;flex-direction:column}
#header{background:#111;padding:.5rem 1rem;border-bottom:1px solid #222;display:flex;align-items:center;gap:1rem}
#header h1{color:#7ef;font-size:1rem}
#apstate{font-size:.85rem;padding:.2rem .6rem;border-radius:4px;background:#1a1a1a;border:1px solid #333}
#body{display:flex;flex:1;overflow:hidden}
#main{flex:1;display:flex;flex-direction:column;padding:.5rem;gap:.5rem}
#streams{display:flex;flex-direction:column;flex:1;gap:.5rem}
#main-stream-wrap{flex:2;position:relative;background:#000;border-radius:6px;overflow:hidden;border:2px solid #333}
#main-stream-wrap img{width:100%;height:100%;object-fit:contain}
#stream-label{position:absolute;top:.4rem;left:.4rem;background:rgba(0,0,0,.7);
  padding:.2rem .5rem;border-radius:4px;font-size:.75rem;color:#7ef;border:1px solid #7ef}
#boundary-wrap{flex:1;position:relative;background:#000;border-radius:6px;overflow:hidden;border:1px solid #222;max-height:220px}
#boundary-wrap img{width:100%;height:100%;object-fit:contain}
#boundary-label{position:absolute;top:.4rem;left:.4rem;background:rgba(0,0,0,.7);
  padding:.2rem .5rem;border-radius:4px;font-size:.7rem;color:#fa0;border:1px solid #fa0}
#sidebar{width:240px;background:#111;border-left:1px solid #222;padding:.75rem;display:flex;flex-direction:column;gap:.75rem;overflow-y:auto}
.card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;padding:.6rem}
.card .lbl{font-size:.65rem;color:#666;text-transform:uppercase;margin-bottom:.2rem}
.card .val{font-size:1rem;font-weight:bold}
.ARMED{color:#4f4}.DISARMED{color:#f44}
.GOOD{color:#4f4}.OK{color:#ff4}.WEAK{color:#f84}.NONE{color:#f44}
.ok{color:#4f4}.bad{color:#f44}
#boundary-info{font-size:.7rem;line-height:1.6}
.side-ok{color:#4f4}.side-no{color:#444}
#dot{width:8px;height:8px;border-radius:50%;background:#4f4;display:inline-block;
  animation:pulse 1s infinite;margin-right:.4rem}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
#url-info{font-size:.65rem;color:#555;line-height:1.8}
</style>
</head>
<body>
<div id="header">
  <span id="dot"></span>
  <h1>ASCEND Autopilot</h1>
  <span id="apstate">IDLE</span>
</div>
<div id="body">
  <div id="main">
    <div id="streams">
      <div id="main-stream-wrap">
        <img id="main-stream" src="" alt="stream">
        <div id="stream-label">RAW CAM</div>
      </div>
      <div id="boundary-wrap">
        <img id="boundary-stream" src="http://PIIP:8084/stream" alt="boundary">
        <div id="boundary-label">BOUNDARY DETECTION</div>
      </div>
    </div>
  </div>
  <div id="sidebar">
    <div class="card">
      <div class="lbl">Mode</div><div class="val" id="mode">—</div>
    </div>
    <div class="card">
      <div class="lbl">Armed</div><div class="val" id="armed">—</div>
    </div>
    <div class="card">
      <div class="lbl">Altitude</div><div class="val" id="alt">—</div>
    </div>
    <div class="card">
      <div class="lbl">Flow Quality</div>
      <div class="val" id="fq">—</div>
      <div id="fv" style="font-size:.7rem;color:#666;margin-top:.2rem"></div>
    </div>
    <div class="card">
      <div class="lbl">EKF</div>
      <div class="ekf" id="ekf" style="font-size:.75rem;line-height:1.8">—</div>
    </div>
    <div class="card">
      <div class="lbl">Autopilot</div>
      <div class="val" id="apstate2">IDLE</div>
    </div>
    <div class="card" id="boundary-card">
      <div class="lbl">Boundary</div>
      <div id="boundary-info">not running</div>
    </div>
    <div class="card">
      <div class="lbl">Streams</div>
      <div id="url-info"></div>
    </div>
  </div>
</div>
<script>
const EKF_BITS=[[0,'ATT'],[1,'VEL_H'],[2,'VEL_V'],[3,'POS_REL'],[5,'POS_Z']];
const STREAM_URLS={
  raw:'http://PIIP:8081/',
  orb:'http://PIIP:8081/',
  qr: 'http://PIIP:8081/',
};
const STREAM_LABELS={raw:'RAW CAM — TAKEOFF',orb:'ORB TRACKING — SURVEY',qr:'QR PRECISION LANDING'};
let lastStream='';

const src=new EventSource('/events');
src.onmessage=e=>{
  try{
    const d=JSON.parse(e.data);
    document.getElementById('mode').textContent=d.mode||'—';
    const a=document.getElementById('armed');
    a.textContent=d.armed?'ARMED':'DISARMED';
    a.className='val '+(d.armed?'ARMED':'DISARMED');
    document.getElementById('alt').textContent=(d.alt_m||0).toFixed(3)+' m ['+d.alt_source+']';
    const f=document.getElementById('fq');
    f.textContent=(d.flow_qual||0)+'/255 ['+d.flow_health+']';
    f.className='val '+d.flow_health;
    document.getElementById('fv').textContent='vx='+(d.flow_vx||0).toFixed(3)+'  vy='+(d.flow_vy||0).toFixed(3);
    document.getElementById('ekf').innerHTML=EKF_BITS.map(([b,n])=>{
      const ok=(d.ekf_flags>>b)&1;
      return '<span class="'+(ok?'ok':'bad')+'">'+(ok?'✓':'✗')+' '+n+'</span>';
    }).join('&nbsp;');
    const ap=d.ap_state||'IDLE';
    document.getElementById('apstate').textContent=ap;
    document.getElementById('apstate2').textContent=ap;

    // Switch main stream — only update src on actual stream change
    const s=d.active_stream||'raw';
    if(s!==lastStream){
      lastStream=s;
      const img=document.getElementById('main-stream');
      // Add timestamp to force browser to reconnect the MJPEG stream
      img.src=STREAM_URLS[s]+'?t='+Date.now();
      document.getElementById('stream-label').textContent=STREAM_LABELS[s]||s;
    }

    // Boundary info
    const b=d.boundary||{};
    if(b.running){
      const sides=b.sides||{};
      const si=Object.entries(sides).map(([k,v])=>
        '<span class="'+(v?'side-ok':'side-no')+'">'+k+'='+v+'</span>').join('  ');
      document.getElementById('boundary-info').innerHTML=
        si+'<br>inside='+b.inside+'  pivot='+b.pivot+'  confirmed='+b.confirmed;
    } else {
      document.getElementById('boundary-info').textContent='not running';
    }

    // URL info
    document.getElementById('url-info').innerHTML=
      'main :8081 (orb_main.py)<br>boundary :8084<br>dashboard :5760';

  }catch(ex){console.error(ex)}
};
src.onerror=()=>{document.getElementById('dot').style.background='#f44'};
</script>
</body>
</html>"""

class _ThreadedHTTP(ThreadingMixIn,HTTPServer):
    allow_reuse_address=True; daemon_threads=True
    def get_request(self):
        r,a=self.socket.accept()
        r.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        return r,a

def _get_ip():
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80))
        ip=s.getsockname()[0]; s.close(); return ip
    except: return 'localhost'

class _DashHandler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass

    def do_GET(self):
        if self.path in ('/','/index.html'):
            ip=_get_ip()
            html=DASHBOARD_HTML.replace('PIIP',ip).encode()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',len(html))
            self.end_headers(); self.wfile.write(html)

        elif self.path=='/events':
            self.send_response(200)
            self.send_header('Content-Type','text/event-stream')
            self.send_header('Cache-Control','no-cache')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            try:
                while True:
                    with _lock:
                        payload={
                            'armed':_armed,'mode':_mode_str,
                            'alt_m':round(_alt,3),'alt_source':_alt_source,
                            'flow_qual':_flow_qual,'flow_health':_flow_health(),
                            'flow_vx':round(_flow_vx,4),'flow_vy':round(_flow_vy,4),
                            'ekf_flags':_ekf_flags,'ap_state':_ap_state_label,
                            'active_stream':_active_stream,
                            'boundary':dict(_boundary_status),
                        }
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush(); time.sleep(0.25)
            except (BrokenPipeError,ConnectionResetError,OSError): pass

        elif self.path=='/status':
            with _lock:
                p={'armed':_armed,'mode':_mode_str,'alt_m':round(_alt,3),'ap_state':_ap_state_label}
            b=json.dumps(p,indent=2).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',len(b)); self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()

def start_dashboard():
    srv=_ThreadedHTTP(('0.0.0.0',DASHBOARD_PORT),_DashHandler)
    threading.Thread(target=srv.serve_forever,daemon=True).start()
    ip=_get_ip()
    print(f"\n[*] Dashboard → http://{ip}:{DASHBOARD_PORT}/")
    print(f"[*] Streams:")
    print(f"    Raw cam  → http://{ip}:{PORT_RAW}/stream")
    print(f"    ORB      → http://{ip}:{PORT_ORB}/stream  (fed by orb_main.py)")
    print(f"    QR land  → http://{ip}:{PORT_QR}/stream")
    print(f"    Boundary → http://{ip}:{PORT_BOUNDARY}/stream\n")

# ─────────────────────────────────────────────────────────────────────────────
# Boundary detection thread (from boundary_core.py logic)
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryThread:
    """
    Runs boundary detection in background.
    Streams annotated frames to PORT_BOUNDARY.
    Publishes status to _boundary_status.
    sim=True → no FC commands sent.
    """
    def __init__(self, sim=True):
        self.sim      = sim
        self._running = False
        self._thread  = None
        self._jpg     = None
        self._jpg_lock= threading.Lock()
        self._srv     = None

    def start(self):
        self._running = True
        self._start_stream_server()
        self._thread = threading.Thread(target=self._run,
                                         name='boundary',daemon=True)
        self._thread.start()
        log(f"[Boundary] started sim={self.sim} stream→:{PORT_BOUNDARY}/stream")

    def stop(self):
        self._running = False
        if self._srv: self._srv.shutdown()
        log("[Boundary] stopped")

    def _start_stream_server(self):
        import http.server, socketserver as _ss
        b = self
        class _H(http.server.BaseHTTPRequestHandler):
            def log_message(self,*a): pass
            def do_GET(self):
                if self.path=='/stream':
                    self.send_response(200)
                    self.send_header('Content-Type','multipart/x-mixed-replace; boundary=--jpgbnd')
                    self.end_headers()
                    try:
                        while b._running:
                            with b._jpg_lock: jpg=b._jpg
                            if jpg is None: time.sleep(0.02); continue
                            self.wfile.write(b'--jpgbnd\r\nContent-Type: image/jpeg\r\n'
                                +f'Content-Length: {len(jpg)}\r\n\r\n'.encode()+jpg+b'\r\n')
                            time.sleep(0.05)
                    except (BrokenPipeError,ConnectionResetError): pass
                else: self.send_response(404); self.end_headers()
        class _S(_ss.ThreadingTCPServer):
            allow_reuse_address=True
        self._srv=_S(('0.0.0.0',PORT_BOUNDARY),_H)
        threading.Thread(target=self._srv.serve_forever,daemon=True).start()

    def _run(self):
        global _boundary_status
        import cv2, numpy as np
        # Import boundary core functions
        try:
            from boundary_core import (detect_yellow_lines, BorderMap,
                                        load_calib, PiCamSource, draw_frame,
                                        draw_debug, BoundaryResult)
        except ImportError as e:
            log(f"[Boundary] boundary_core import failed: {e}"); return

        cfg  = load_calib()
        bmap = BorderMap()
        cap  = PiCamSource(width=640, height=480, cam_index=1)

        frame_n = 0
        last_result = BoundaryResult()
        last_lines  = []
        last_mask   = None

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None: time.sleep(0.02); continue
            frame_n += 1
            fh, fw = frame.shape[:2]

            n = max(1, cfg.get('PROCESS_EVERY_N', 2))
            if frame_n % n == 0:
                scale = max(0.25, min(1.0, float(cfg.get('DETECT_SCALE', 0.5))))
                lines, ymask = detect_yellow_lines(frame, cfg, scale=scale)
                bmap.update(lines, fw, fh)
                last_result = bmap.build_result(fw, fh)
                last_lines  = lines
                last_mask   = ymask

                # Update global boundary status
                _boundary_status = {
                    'running'  : True,
                    'sides'    : last_result.sides,
                    'inside'   : last_result.inside_yellow,
                    'pivot'    : last_result.centre_touched,
                    'confirmed': last_result.confirmed_sides,
                }

                # Send FC commands if not sim
                if not self.sim:
                    self._execute(last_result)

            # Draw and push to stream
            fps = 0.0
            disp = draw_frame(frame, last_lines, bmap, last_result, fps,
                               last_mask if last_mask is not None
                               else np.zeros((fh,fw),dtype=np.uint8), cfg)
            _, jpg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 65])
            with self._jpg_lock: self._jpg = jpg.tobytes()

        cap.release()
        _boundary_status = {'running': False}

    def _execute(self, result):
        """Send FC commands based on boundary result."""
        if result.centre_touched:
            _stop_hold(); _start_hold()
            log("[Boundary] CENTRE TOUCHED — pivot signal")
            return
        if result.inside_yellow and result.rect is not None:
            sides=result.sides; speed=0.15; vx=vy=0.0
            if sides.get('TOP'):    vx= speed
            elif sides.get('BOTTOM'): vx=-speed
            if sides.get('LEFT'):   vy= speed
            elif sides.get('RIGHT'): vy=-speed
            cmd_velocity(vx,vy,0.0)
            return
        if result.valid() and result.centre_err:
            ex,ey=result.centre_err; speed=0.4
            if abs(ex)>abs(ey): vy=math.copysign(speed,ex); vx=0.0
            elif abs(ey)>35:    vx=math.copysign(speed,ey); vy=0.0
            else: vx=vy=0.0
            cmd_velocity(vx,vy,0.0)

_boundary_thread = None

# ─────────────────────────────────────────────────────────────────────────────
# Hold system
# ─────────────────────────────────────────────────────────────────────────────

_hold_active=False; _hold_thread=None; _hold_z=None
_hold_continuous=False; _hold_cont_dir=None; _hold_cont_spd=0.0
_hold_cont_vec=(0.0,0.0,0.0); _hold_stop_evt=threading.Event()

def _body_to_ned_vel(dx,dy,speed):
    with _lock: yaw=_yaw
    cy=math.cos(yaw); sy=math.sin(yaw)
    return (cy*dx-sy*dy)*speed,(sy*dx+cy*dy)*speed

def _hold_loop():
    while not _hold_stop_evt.is_set():
        if not master: time.sleep(0.1); continue
        with _lock: is_cont=_hold_continuous; vec=_hold_cont_vec; spd=_hold_cont_spd
        if is_cont:
            dx_u=vec[0]/(spd or 1.0); dy_u=vec[1]/(spd or 1.0)
            vn,ve=_body_to_ned_vel(dx_u,dy_u,spd); vd=vec[2]
            master.mav.set_position_target_local_ned_send(
                0,master.target_system,master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,0b0000111111000111,
                0,0,0,vn,ve,vd,0,0,0,0,0)
            _hold_stop_evt.wait(timeout=0.1)
        else:
            master.mav.set_position_target_local_ned_send(
                0,master.target_system,master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,0b0000111111000111,
                0,0,0,0.0,0.0,0.0,0,0,0,0,0)
            _hold_stop_evt.wait(timeout=0.2)

def _stop_hold():
    global _hold_active,_hold_thread,_hold_continuous,_hold_cont_dir,_hold_cont_spd,_hold_cont_vec
    _hold_stop_evt.set()
    if _hold_thread and _hold_thread.is_alive(): _hold_thread.join(timeout=0.5)
    _hold_active=False; _hold_continuous=False; _hold_cont_dir=None
    _hold_cont_spd=0.0; _hold_cont_vec=(0.0,0.0,0.0)

def _start_hold(ned_z=None):
    global _hold_active,_hold_thread,_hold_z,_hold_continuous
    _stop_hold(); _hold_stop_evt.clear()
    _hold_z=ned_z; _hold_active=True; _hold_continuous=False
    _hold_thread=threading.Thread(target=_hold_loop,daemon=True); _hold_thread.start()

def _start_continuous(dx,dy,dz,speed,direction_str):
    global _hold_active,_hold_thread,_hold_z,_hold_continuous,_hold_cont_dir,_hold_cont_spd,_hold_cont_vec
    _stop_hold(); _hold_stop_evt.clear()
    _hold_z=None; _hold_continuous=True; _hold_cont_dir=direction_str
    _hold_cont_spd=speed; _hold_cont_vec=(dx*speed,dy*speed,dz*speed)
    _hold_active=True
    _hold_thread=threading.Thread(target=_hold_loop,daemon=True); _hold_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# Action system
# ─────────────────────────────────────────────────────────────────────────────

_action_thread=None; _action_cancel=threading.Event()

def _cancel_action(wait=True):
    global _action_thread
    _action_cancel.set()
    if wait and _action_thread and _action_thread.is_alive():
        _action_thread.join(timeout=1.0)

def _run_action(fn,*args):
    global _action_thread
    _cancel_action(wait=True); _action_cancel.clear()
    _action_thread=threading.Thread(target=fn,args=args,daemon=True)
    _action_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# FC commands
# ─────────────────────────────────────────────────────────────────────────────

def set_mode(mode_name):
    mid=MODES.get(mode_name.lower())
    if mid is None: print(f"[!] Unknown mode '{mode_name}'"); return
    master.mav.command_long_send(
        master.target_system,master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,mid,0,0,0,0,0)
    log(f"Mode → {mode_name.upper()}")

def cmd_stop():
    _cancel_action(wait=False); _start_hold()
    log(f"STOP — soft hold at {_alt:.2f}m")

def cmd_arm():
    set_mode('guided'); time.sleep(0.5)
    master.mav.command_long_send(
        master.target_system,master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,1,0,0,0,0,0,0)
    deadline=time.time()+10
    while time.time()<deadline:
        with _lock:
            if _armed: break
        time.sleep(0.05)
    log("ARMED")

def cmd_velocity(vx,vy,vz=0.0):
    _stop_hold()
    spd=math.hypot(vx,vy)
    if spd>1e-6: vn,ve=_body_to_ned_vel(vx/spd,vy/spd,spd)
    else: vn,ve=0.0,0.0
    master.mav.set_position_target_local_ned_send(
        0,master.target_system,master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,0b0000111111000111,
        0,0,0,vn,ve,float(vz),0,0,0,0,0)

def _cmd_takeoff_blocking(alt):
    global _home_heading, _active_stream
    target=float(alt); set_mode('guided'); time.sleep(0.3)
    with _lock: _home_heading=math.degrees(_yaw)%360
    log(f"HOME HEADING: {_home_heading:.1f}°")
    _active_stream = 'raw'
    master.mav.command_long_send(
        master.target_system,master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,0,target)
    log(f"TAKEOFF → {target}m")
    deadline=time.time()+30
    while time.time()<deadline and not _action_cancel.is_set():
        print(f"    alt:{_alt:.2f}/{target}m  flow:{_flow_qual}/255    ",end='\r',flush=True)
        if abs(_alt-target)<0.05:
            print()
            log("Reached — settling 3s...")
            settle=time.time()+3.0
            while time.time()<settle:
                if _action_cancel.is_set(): return
                time.sleep(0.1)
            with _lock: sh=math.degrees(_yaw)%360
            err=(_home_heading-sh+360)%360
            if err>180: err-=360
            d='right' if err>0 else 'left'
            log(f"HEADING err={err:+.1f}° → turning {d}")
            if abs(err)>3.0: _cmd_turn_blocking(d,abs(err),30.0)
            _start_hold(); return
        time.sleep(0.05)
    if _action_cancel.is_set(): print(); log("TAKEOFF OVERRIDDEN"); return
    print(f"\n[!] Takeoff timeout at {_alt:.2f}m")

def cmd_takeoff(alt): _run_action(_cmd_takeoff_blocking,alt)

def cmd_land():
    global _active_stream
    _cancel_action(wait=False); _stop_hold()
    master.mav.command_long_send(
        master.target_system,master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,0,0,0,0,0,0,0,0)
    log("LAND")

def cmd_rtl():
    _cancel_action(wait=False); _stop_hold(); set_mode('rtl'); log("RTL")

def cmd_disarm():
    master.mav.command_long_send(
        master.target_system,master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,0,0,0,0,0,0,0)
    log("DISARMED")

def _cmd_move_blocking(parts):
    if len(parts)<3: print("[!] move <dir> <dist|spd> [c]"); return
    direction=parts[1]
    if direction not in MOVE_DIRECTIONS: print(f"[!] Unknown dir '{direction}'"); return
    dx,dy,dz=MOVE_DIRECTIONS[direction]
    if len(parts)>3 and parts[3]=='c':
        speed=float(parts[2]); _start_continuous(dx,dy,dz,speed,direction)
        log(f"MOVE {direction.upper()} CONT @ {speed}m/s"); return
    if len(parts)>3 and parts[3]=='spd':
        speed=float(parts[2]); _stop_hold()
        cmd_velocity(dx*speed,dy*speed,dz*speed); return
    dist=float(parts[2]); v_target=0.15; Kp=0.8; v_min=0.05; v_max=0.30
    timeout=(dist/v_target)*3.0; _stop_hold()
    log(f"MOVE {direction} {dist*100:.0f}cm")
    cmd_velocity(dx*v_target,dy*v_target,dz*v_target)
    travelled=0.0; t_start=time.time(); _prev=(None,None)
    while not _action_cancel.is_set():
        time.sleep(0.05); elapsed=time.time()-t_start
        with _lock: vx=_flow_vx; vy=_flow_vy; fq=_flow_qual
        cur=(vx,vy)
        if cur!=_prev:
            v_act=vx*dx+vy*dy; travelled+=abs(v_act*0.05); _prev=cur
            if fq>=50:
                v_cmd=max(v_min,min(v_max,v_target+Kp*(v_target-abs(v_act))))
                cmd_velocity(dx*v_cmd,dy*v_cmd,dz*v_cmd)
        if travelled>=dist: log(f"MOVE done {travelled*100:.1f}cm"); break
        if elapsed>=timeout: log(f"MOVE timeout {travelled*100:.1f}cm"); break
    if _action_cancel.is_set(): log("MOVE OVERRIDDEN"); return
    _start_hold()

def cmd_move(parts): _run_action(_cmd_move_blocking,parts)

def _cmd_turn_blocking(direction,degrees,speed):
    _stop_hold()
    master.mav.command_long_send(
        master.target_system,master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,0,
        degrees,speed,TURN_DIR[direction],1,0,0,0)
    log(f"TURN {direction.upper()} {degrees:.0f}° @ {speed:.0f}°/s")
    with _lock: start_yaw=_yaw
    target_delta=math.radians(degrees)
    deadline=time.time()+(degrees/speed)*1.5+1.0
    while not _action_cancel.is_set() and time.time()<deadline:
        time.sleep(0.05)
        with _lock: cur=_yaw
        delta=abs(math.atan2(math.sin(cur-start_yaw),math.cos(cur-start_yaw)))
        if delta>=target_delta-math.radians(3): break
    if _action_cancel.is_set(): log("TURN OVERRIDDEN"); return
    log("TURN done"); _start_hold()

def cmd_turn(parts):
    if len(parts)<2 or parts[1] not in TURN_DIR:
        print("[!] turn <left|right> [deg]"); return
    _run_action(_cmd_turn_blocking,parts[1],float(parts[2]) if len(parts)>2 else 90.0,30.0)

def _cmd_goto_blocking(target_alt):
    with _lock: current=_alt
    v_sign=-1.0 if target_alt>current else 1.0
    _stop_hold(); log(f"GOTO {target_alt:.2f}m")
    deadline=time.time()+(abs(target_alt-current)/0.3)*2.5+2.0
    while not _action_cancel.is_set() and time.time()<deadline:
        master.mav.set_position_target_local_ned_send(
            0,master.target_system,master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,0b0000111111000111,
            0,0,0,0.0,0.0,v_sign*0.3,0,0,0,0,0)
        time.sleep(0.1)
        with _lock: cur=_alt
        print(f"    alt:{cur:.2f}/{target_alt:.2f}m    ",end='\r',flush=True)
        if abs(cur-target_alt)<0.08:
            print(); log(f"GOTO reached {cur:.2f}m"); _start_hold(); return
    print()
    if _action_cancel.is_set(): log(f"GOTO OVERRIDDEN"); return
    log(f"GOTO timeout"); _start_hold()

def cmd_goto(parts):
    if len(parts)<2: print("[!] goto <metres>"); return
    try: t=float(parts[1])
    except: print("[!] number"); return
    if t<0.2: print("[!] min 0.2m"); return
    _run_action(_cmd_goto_blocking,t)

def cmd_status():
    with _lock:
        armed=_armed; mode=_mode_str; alt=_alt; src=_alt_source
        fq=_flow_qual; fvx=_flow_vx; fvy=_flow_vy; ekf=_ekf_flags
    print(f"  MODE : {mode}  ({'ARMED' if armed else 'DISARMED'})")
    print(f"  ALT  : {alt:.2f}m  [{src}]")
    print(f"  FLOW : {fq}/255 [{_flow_health()}]  vx={fvx:.3f} vy={fvy:.3f}")
    print(f"  EKF  : {_ekf_label(ekf)}")
    print(f"  AP   : {_ap_state_label}")
    print(f"  BNDRY: {_boundary_status}")

def cmd_monitor():
    print("[*] Monitor — Ctrl+C to return\n")
    try:
        while True:
            with _lock:
                armed=_armed; mode=_mode_str; alt=_alt; src=_alt_source
                fq=_flow_qual; fvx=_flow_vx; fvy=_flow_vy; ekf=_ekf_flags
            lines=[
                f"  ┌──────────── ASCEND ────────────┐",
                f"  │ MODE  {mode:<10} {'[ARMED]' if armed else '[DISARM]'}      │",
                f"  │ ALT   {alt:.3f}m  [{src}]           │",
                f"  │ FLOW  {fq:>3}/255 [{_flow_health():<4}] vx={fvx:+.3f}    │",
                f"  │ EKF   {_ekf_label(ekf):<30} │",
                f"  │ AP    {_ap_state_label:<30} │",
                f"  └────────────────────────────────┘",
            ]
            print('\033[F'*7 if hasattr(cmd_monitor,'_s') else '',end='')
            cmd_monitor._s=True
            print('\n'.join(lines),flush=True); time.sleep(0.2)
    except KeyboardInterrupt:
        if hasattr(cmd_monitor,'_s'): del cmd_monitor._s
        print("\n[*] Monitor exited.")

def cmd_tune():
    def _sp(n,v):
        master.mav.param_set_send(master.target_system,master.target_component,
            n.encode(),float(v),mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        ack=master.recv_match(type='PARAM_VALUE',blocking=True,timeout=2)
        print(f"  {n} = {ack.param_value if ack else '?'}")
    _sp('FHLD_XY_P',0.35); _sp('FHLD_BRAKE_RATE',10.0); _sp('EK3_FLOW_M_NSE',0.15)

# ─────────────────────────────────────────────────────────────────────────────
# QR lander
# ─────────────────────────────────────────────────────────────────────────────

class QRLander:
    def __init__(self,seed_path,cam_index=0):
        self.centred=False; self.found=False
        self._running=True; self._seed=seed_path; self._cam=cam_index
        self._thread=threading.Thread(target=self._run,name='qr-land',daemon=True)

    def start(self): self._thread.start(); log("[QR] Lander started")
    def stop(self):  self._running=False

    def _run(self):
        try:
            from qr_land import SeedMatcher,MarkerWorker,PiCamSource
        except ImportError:
            log("[QR] qr_land.py not found"); return
        sm=SeedMatcher(); ok,msg=sm.load(self._seed)
        if not ok: log(f"[QR] Seed failed: {msg}"); return
        cam=PiCamSource(cam_index=self._cam)
        worker=MarkerWorker(sm)
        if sm.seed_dict_id: worker.lock_dict(sm.seed_dict_id)
        import numpy as np
        while self._running:
            frame=cam.read()
            if frame is None: time.sleep(0.02); continue
            worker.submit(frame); result=worker.get()
            if result:
                label,corners,angle_diff,angle_ok,msize=result
                fh,fw=frame.shape[:2]
                cx=int(np.mean(corners[:,0])); cy=int(np.mean(corners[:,1]))
                ex=abs(cx-fw//2); ey=abs(cy-fh//2)
                self.found=True
                self.centred=(ex<QR_CENTRE_THRESH and ey<QR_CENTRE_THRESH and angle_ok)
                if not self.centred:
                    nx=(cx-fw/2)/(fw/2); ny=(cy-fh/2)/(fh/2)
                    cmd_velocity(ny*0.15,nx*0.15,0.0)
            else:
                self.found=False; self.centred=False
            time.sleep(0.03)
        cam.release(); worker.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ORB result reader
# ─────────────────────────────────────────────────────────────────────────────

class ORBReader:
    def __init__(self):
        self._result=None; self._lock=threading.Lock(); self._running=True
        threading.Thread(target=self._run,name='orb-reader',daemon=True).start()
    def get(self):
        with self._lock: return self._result
    def stop(self): self._running=False
    def _run(self):
        p=Path('/tmp/orb_result.json'); last=0.0
        while self._running:
            try:
                if p.exists():
                    mt=p.stat().st_mtime
                    if mt!=last:
                        last=mt
                        with self._lock: self._result=json.loads(p.read_text())
            except Exception: pass
            time.sleep(0.05)

# ─────────────────────────────────────────────────────────────────────────────
# Autopilot mission
# ─────────────────────────────────────────────────────────────────────────────

class APState(Enum):
    IDLE=auto(); TAKEOFF=auto(); SURVEY=auto(); LOCKED=auto()
    ACQUIRE=auto(); RETURN=auto(); LAND=auto(); DONE=auto()

_ap_running=False; _ap_cancel=threading.Event(); _ap_thread=None
_orb_reader=ORBReader(); _qr_lander=None

def _set_ap(s:APState):
    global _ap_state_label; _ap_state_label=s.name; log(f"── AP: {s.name}")

def _ap_wait(secs,msg='')->bool:
    if msg: log(msg)
    deadline=time.time()+secs
    while time.time()<deadline:
        if _ap_cancel.is_set(): return False
        time.sleep(0.1)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Serpentine helpers
# ─────────────────────────────────────────────────────────────────────────────

_CAM_HFOV_DEG = 45.0   # IMX296 horizontal FOV
_CAM_OVERLAP  = 0.80   # 80% coverage = 20% overlap per lane

def _lane_shift_m() -> float:
    """Lane shift = ground width * overlap. Computed from live altitude."""
    with _lock: alt = _alt
    if alt < 0.2: alt = 0.9
    gw = 2.0 * alt * math.tan(math.radians(_CAM_HFOV_DEG / 2.0))
    return round(gw * _CAM_OVERLAP, 2)

def _do_serpentine_turn(turn_dir: str):
    """
    90° turn → lane shift → 90° turn (same dir).
    Lane shift distance computed from live altitude + camera FOV.
    """
    shift = _lane_shift_m()
    dur   = shift / 0.3
    log(f"[SERPENTINE] {turn_dir.upper()}  shift={shift:.2f}m  dur={dur:.1f}s  alt={_alt:.2f}m")

    _action_cancel.clear()
    _cmd_turn_blocking(turn_dir, 90.0, 30.0)
    if _ap_cancel.is_set(): return

    _start_continuous(1, 0, 0, 0.3, 'fwd')
    deadline = time.time() + dur
    while time.time() < deadline:
        if _ap_cancel.is_set(): _stop_hold(); return
        time.sleep(0.05)
    _stop_hold(); time.sleep(0.2)

    _action_cancel.clear()
    _cmd_turn_blocking(turn_dir, 90.0, 30.0)
    log("[SERPENTINE] done — resuming survey")


def _ap_mission():
    """
    Mission sequence
    ────────────────
    1. Takeoff to 0.9m
    2. Serpentine survey — fwd @ CRUISE_SPEED
       • boundary_core runs as background thread (sim=True → flags only, no FC cmds)
       • TOP border hit  → serpentine RIGHT (odd passes) / LEFT (even passes)
       • BOTTOM border ignored (came from there)
       • ORB running throughout via orb_main → /tmp/orb_result.json
       • ORB lock → slow to SLOW_SPEED, collect candidates
       • centred_confirmed → stop, screenshot, resume
       • Runs until abort — no hard time limit
    3. abort / qrland to end
    """
    global _ap_running, _active_stream, _boundary_thread

    _ap_cancel.clear()

    # TAKEOFF
    _set_ap(APState.TAKEOFF); _active_stream = 'raw'
    _action_cancel.clear()
    _cmd_takeoff_blocking(0.9)
    if _ap_cancel.is_set(): _smooth_abort(); return
    log("TAKEOFF — 0.9m")

    # Start boundary in flag mode (sim=True → no FC commands from boundary)
    if _boundary_thread is None:
        _boundary_thread = BoundaryThread(sim=True)
        _boundary_thread.start()
        log("[Boundary] flag mode started")
    time.sleep(1.0)   # let boundary settle

    # SURVEY
    _set_ap(APState.SURVEY); _active_stream = 'orb'

    CRUISE_SPEED = 0.1
    SLOW_SPEED   = 0.05
    turn_dirs    = ['right', 'left']
    turn_idx     = 0
    top_was_hit  = False
    screenshot_taken = False
    slow_was     = False

    _start_continuous(1, 0, 0, CRUISE_SPEED, 'fwd')
    log(f"SURVEY — fwd @ {CRUISE_SPEED}m/s  serpentine on TOP border")

    while not _ap_cancel.is_set():

        result  = _orb_reader.get()
        bstatus = dict(_boundary_status)

        # ── ORB speed control ─────────────────────────────────────────────────
        slow_now = bool(result and result.get('slow_down'))
        if slow_now and not slow_was:
            _start_continuous(1, 0, 0, SLOW_SPEED, 'fwd')
            log("SLOW — ORB lock")
        elif not slow_now and slow_was:
            _start_continuous(1, 0, 0, CRUISE_SPEED, 'fwd')
            log("CRUISE — lock gone")
        slow_was = slow_now

        # ── Screenshot on centred_confirmed ───────────────────────────────────
        if result and result.get('centred_confirmed') and not screenshot_taken:
            log("★ CENTRED CONFIRMED — screenshot")
            _stop_hold(); time.sleep(0.3)
            import shutil as _sh
            snap = Path('/tmp/orb_snapshot.jpg')
            if snap.exists():
                ts  = datetime.datetime.now().strftime('%H%M%S')
                dst = Path(f'acquired_targets/target_{ts}.jpg')
                Path('acquired_targets').mkdir(exist_ok=True)
                _sh.copy(str(snap), str(dst))
                log(f"Screenshot → {dst}")
            screenshot_taken = True
            _start_continuous(1, 0, 0, SLOW_SPEED if slow_now else CRUISE_SPEED, 'fwd')

        # ── TOP border → serpentine turn ──────────────────────────────────────
        top_now = (bstatus.get('running') and
                   bstatus.get('sides', {}).get('TOP', 0) == 1 and
                   bstatus.get('inside', False))

        if top_now and not top_was_hit:
            turn_dir  = turn_dirs[turn_idx % 2]
            turn_idx += 1
            log(f"[BORDER] TOP — serpentine {turn_dir.upper()} pass {turn_idx}")
            _stop_hold()
            screenshot_taken = False
            _do_serpentine_turn(turn_dir)
            if _ap_cancel.is_set(): _smooth_abort(); return
            _start_continuous(1, 0, 0, CRUISE_SPEED, 'fwd')
            slow_was = False

        top_was_hit = top_now
        time.sleep(0.1)

    _stop_hold()
    _smooth_abort()
    global _ap_running; _ap_running = False; _active_stream = 'raw'

def _run_qr_land():
    _set_ap(APState.LAND)
    if _qr_lander and _qr_lander.found:
        deadline=time.time()+QR_LAND_TIMEOUT
        while time.time()<deadline:
            if _ap_cancel.is_set(): break
            if _qr_lander.centred:
                cmd_land(); log("LAND sent"); time.sleep(5); return
            time.sleep(0.1)
        log("[QR] Timeout — force land")
    cmd_land(); time.sleep(5)

def _smooth_abort():
    global _ap_running
    log("AP ABORT — smooth stop"); _stop_hold()
    _start_hold(); time.sleep(1.0); cmd_land()
    _ap_running=False; _set_ap(APState.IDLE)

def cmd_start():
    global _ap_running,_ap_thread
    if _ap_running: print("[!] Already running — type abort"); return
    _ap_running=True
    _ap_thread=threading.Thread(target=_ap_mission,name='autopilot',daemon=True)
    _ap_thread.start()
    log("Autopilot started")

def cmd_abort():
    global _ap_running
    log("ABORT"); _ap_cancel.set(); _action_cancel.set()
    _stop_hold(); time.sleep(0.3); _start_hold(); time.sleep(1.0)
    cmd_land(); _ap_running=False; _set_ap(APState.IDLE)

def cmd_qrland():
    global _qr_lander, _active_stream
    log("[QR] Manual QR landing")
    # Start lander if not already running
    if _qr_lander is None:
        seed = Path(QR_SEED)
        if not seed.is_file():
            log(f"[QR] Seed not found: {QR_SEED} — use setqr <path>")
            return
        _qr_lander = QRLander(str(seed), QR_CAM)
        _qr_lander.start()
        log(f"[QR] Lander started with seed={QR_SEED}")
        time.sleep(2.0)   # give it time to detect
    _active_stream = 'qr'
    _run_qr_land()

def cmd_setqr(parts):
    global QR_SEED, _qr_lander
    if len(parts) < 2:
        print(f"  current QR seed: {QR_SEED}"); return
    path = parts[1]
    if not Path(path).is_file():
        print(f"[!] File not found: {path}"); return
    QR_SEED = path
    # Restart lander with new seed
    if _qr_lander:
        _qr_lander.stop()
        _qr_lander = None
    log(f"[QR] Seed set → {QR_SEED}")

def cmd_boundary(parts):
    global _boundary_thread
    if len(parts)<2: print("[!] boundary sim|real|stop"); return
    sub=parts[1]
    if sub=='stop':
        if _boundary_thread: _boundary_thread.stop(); _boundary_thread=None
        return
    sim=(sub=='sim')
    if _boundary_thread: _boundary_thread.stop()
    _boundary_thread=BoundaryThread(sim=sim)
    _boundary_thread.start()

_centerer = None

def cmd_center(parts):
    global _centerer
    sub = parts[1] if len(parts) > 1 else 'start'

    if sub == 'stop':
        if _centerer: _centerer.stop(); _centerer = None
        log("[Center] stopped")
        return

    if sub == 'status':
        if _centerer: print(f"  {_centerer.status}")
        else: print("  not running")
        return

    # start — read latest ORB result for snapshot + seed name
    result = _orb_reader.get()
    if result is None or not result.get('confirmed'):
        print("[!] No ORB lock — cannot center"); return

    lock_box  = result.get('lock_box')
    seed_name = Path(result.get('active_seed','target')).stem if result.get('active_seed') else 'target'

    if lock_box is None:
        print("[!] No lock box in ORB result"); return

    # Read snapshot from Pi seed folder via orb_main's camera
    # snapshot is stored in /tmp/orb_snapshot.jpg by orb_main
    import cv2 as _cv2
    snap_path = Path('/tmp/orb_snapshot.jpg')
    if snap_path.exists():
        snapshot = _cv2.imread(str(snap_path))
    else:
        snapshot = None
        log("[Center] no snapshot file — using None")

    from centering import Centerer, CenterState
    _centerer = Centerer(
        send_velocity_fn = cmd_velocity,
        hover_fn         = cmd_stop,
        seed_name        = seed_name,
    )

    fw = 640; fh = 480  # default frame size
    _centerer.start(snapshot, fw, fh)
    log(f"[Center] started  seed={seed_name}")

    # Run centering loop in background
    def _center_loop():
        global _centerer
        while _centerer and _centerer._running:
            r = _orb_reader.get()
            if r:
                box = r.get('lock_box')
                state = _centerer.update(box, fw, fh)
                if state == CenterState.CONFIRMED:
                    log("★ TARGET CONFIRMED ★")
                    break
                elif state == CenterState.FAILED:
                    log("[Center] failed — lock lost")
                    break
            time.sleep(0.1)

    threading.Thread(target=_center_loop, name='center-loop', daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Connect
# ─────────────────────────────────────────────────────────────────────────────

def connect():
    global master
    print(f"[*] Connecting {CONNECTION} @ {BAUD_RATE}...")
    master=mavutil.mavlink_connection(CONNECTION,baud=BAUD_RATE)
    master.wait_heartbeat()
    print(f"[+] Connected — sys {master.target_system}")
    threading.Thread(target=telemetry_loop,daemon=True).start()
    threading.Timer(5.0,fetch_params_background).start()

# ─────────────────────────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────────────────────────

HELP="""
━━━━━━━━━━━━━━━━  ASCEND REPL  ━━━━━━━━━━━━━━━━
Autopilot:
  start              begin mission
  abort              smooth stop + land
  qrland             QR precision landing now
  setqr <path>       set QR seed image (default: seedqr.jpeg)
  boundary sim       start boundary (sim, no FC)
  boundary real      start boundary (real FC)
  boundary stop      stop boundary
  center             centre drone on current ORB lock (4s confirm)
  center stop        abort centering
  center status      centering state

FC / Manual:
  arm                guided + arm
  takeoff <m>        climb
  land / x           land now
  stop               soft hover
  rtl                return to launch
  disarm             force disarm
  move <dir> <dist|spd> [c]
  turn <left|right> [deg]
  goto <m>           change altitude
  mode <name>

Info:
  status / monitor / tune / help / quit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def main():
    connect()
    start_dashboard()
    print(HELP)
    while True:
        try: raw=input("ascend> ").strip()
        except (EOFError,KeyboardInterrupt):
            print("\n[*] Exiting"); cmd_land(); break
        if not raw: continue
        parts=raw.lower().split(); cmd=parts[0]
        if   cmd in ('quit','exit'):   cmd_land(); break
        elif cmd=='help':              print(HELP)
        elif cmd=='start':             cmd_start()
        elif cmd=='abort':             cmd_abort()
        elif cmd=='qrland':            cmd_qrland()
        elif cmd=='setqr':             cmd_setqr(parts)
        elif cmd=='boundary':          cmd_boundary(parts)
        elif cmd=='center':            cmd_center(parts)
        elif cmd=='arm':               cmd_arm()
        elif cmd=='takeoff':           cmd_takeoff(float(parts[1]) if len(parts)>1 else 1.0)
        elif cmd in ('land','x'):      cmd_land()
        elif cmd=='stop':              cmd_stop()
        elif cmd=='rtl':               cmd_rtl()
        elif cmd=='disarm':            cmd_disarm()
        elif cmd in ('move','mv'):     cmd_move(parts)
        elif cmd=='turn':              cmd_turn(parts)
        elif cmd=='goto':              cmd_goto(parts)
        elif cmd=='status':            cmd_status()
        elif cmd=='monitor':           cmd_monitor()
        elif cmd=='tune':              cmd_tune()
        elif cmd=='mode':
            if len(parts)>1: _cancel_action(wait=False); set_mode(parts[1])
            else: print("[!] mode <name>")
        else: print(f"[!] Unknown: '{cmd}' — type help")

if __name__=='__main__':
    main()
