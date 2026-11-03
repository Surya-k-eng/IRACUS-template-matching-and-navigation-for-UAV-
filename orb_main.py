"""
orb_main.py
═══════════
Two modes only:

  bench-live   Run ORBLocker on live camera (needs orb_core + ROS/picam)
  offload      Pi = camera only, streams to PC via TCP (needs NOTHING extra)

Usage
─────
  # offload mode (Pi side) — no yaml, no orb_core needed
  python3 orb_main.py --mode offload --cam 0

  # bench-live (needs orb_core.py present)
  python3 orb_main.py --mode bench-live --config orb.yaml --source picam
"""

import argparse
import http.server
import json
import logging
import os
import socket
import socketserver
import struct
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("orb_main")

# ── protocol ──────────────────────────────────────────────────────────────────
MSG_FRAME      = 0x01
MSG_RESULT     = 0x02
MSG_PING       = 0x03
MSG_PONG       = 0x04
MSG_SEED_START = 0x05
MSG_SEED_FILE  = 0x06
MSG_SEED_DONE  = 0x07

PI_SEEDS_DIR   = Path('/home/admin/icarusreborn/seeds')

OFFLOAD_DEFAULT_PORT   = 5700
OFFLOAD_DISCOVERY_PORT = 5701
DISCOVERY_MAGIC        = b'ORB_OFFLOAD_HERE'
DISCOVERY_ACK          = b'ORB_PI_ACK'
OFFLOAD_PING_INTERVAL  = 2.0


# ── TCP helpers ───────────────────────────────────────────────────────────────

def _recv_all(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return buf

def _recv_msg(sock):
    hdr = _recv_all(sock, 5)
    msg_type, length = struct.unpack('!BI', hdr)
    payload = _recv_all(sock, length) if length else b''
    return msg_type, payload

def _send_msg(sock, msg_type, payload=b''):
    sock.sendall(struct.pack('!BI', msg_type, len(payload)) + payload)


# ── PiCam ─────────────────────────────────────────────────────────────────────

class PiCamCapture:
    def __init__(self, cam_index=0, width=640, height=480):
        from picamera2 import Picamera2
        self._cam = Picamera2(cam_index)
        cfg = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(0.5)
        log.info(f"[picam] opened index={cam_index} {width}x{height}")

    def read(self):
        try:
            return True, cv2.cvtColor(self._cam.capture_array(), cv2.COLOR_RGB2BGR)
        except Exception as e:
            log.warning(f"[picam] read error: {e}")
            return False, None

    def isOpened(self):
        return True

    def release(self):
        try: self._cam.stop(); self._cam.close()
        except Exception: pass


# ── WebStreamer (annotated MJPEG display on :8081) ────────────────────────────

class WebStreamer:
    def __init__(self, port=8081):
        self._jpg     = None
        self._lock    = threading.Lock()
        self._running = True
        self._actions = []
        streamer = self

        class _H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                if self.path == '/':
                    body = (b'<!DOCTYPE html><html><head><title>ORB</title>'
                            b'<style>body{background:#111;color:#0f0;text-align:center}'
                            b'img{max-width:100%;border:2px solid #0f0}</style></head>'
                            b'<body><img src="/stream"><br>'
                            b'<a href="/quit" style="color:#f44">Quit</a></body></html>')
                    self.send_response(200)
                    self.send_header('Content-Type','text/html')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers(); self.wfile.write(body)
                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type','multipart/x-mixed-replace; boundary=--jpgbnd')
                    self.send_header('Cache-Control','no-cache')
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
                elif self.path == '/quit':
                    streamer._actions.append('quit')
                    self.send_response(200); self.end_headers()
                    self.wfile.write(b'ok')
                else:
                    self.send_response(404); self.end_headers()

        class _S(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        self._server = _S(('0.0.0.0', port), _H)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80)); ip = s.getsockname()[0]; s.close()
        except Exception: ip = 'localhost'
        log.info(f"[WebStreamer] http://{ip}:{port}/")

    def show(self, frame):
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self._lock: self._jpg = jpg.tobytes()

    def waitKey(self):
        if self._actions:
            a = self._actions.pop(0)
            return ord('q') if a == 'quit' else -1
        time.sleep(0.001); return -1

    def stop(self):
        self._running = False
        self._server.shutdown()


# ── UDP discovery ─────────────────────────────────────────────────────────────

def _discover_pc(discovery_port=OFFLOAD_DISCOVERY_PORT, timeout=120.0):
    """Listen for UDP beacon from PC, reply ACK, return PC IP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', discovery_port))
    sock.settimeout(1.0)
    log.info(f"[Discovery] waiting for PC beacon on UDP :{discovery_port} ...")
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            data, addr = sock.recvfrom(256)
        except socket.timeout:
            continue
        if data == DISCOVERY_MAGIC:
            pc_ip = addr[0]
            sock.sendto(DISCOVERY_ACK, addr)
            sock.close()
            log.info(f"[Discovery] PC found at {pc_ip}")
            return pc_ip
    sock.close()
    raise TimeoutError("No PC beacon received")


# ── OffloadLink ───────────────────────────────────────────────────────────────

class OffloadLink:
    """Pi connects OUT to PC's TCP server."""

    def __init__(self, pc_ip, tcp_port=OFFLOAD_DEFAULT_PORT):
        self._stop      = threading.Event()
        self._result    = None
        self._res_lock  = threading.Lock()
        self._conn_lock = threading.Lock()

        log.info(f"[OffloadLink] connecting to {pc_ip}:{tcp_port} ...")
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((pc_ip, tcp_port))
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._conn = conn
        log.info("[OffloadLink] connected")

        threading.Thread(target=self._recv_loop, name='offload-recv').start()
        threading.Thread(target=self._ping_loop, name='offload-ping').start()

    def send_frame(self, bgr, frame_idx, quality=75):
        with self._conn_lock: conn = self._conn
        if conn is None: return
        _, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        payload = struct.pack('!I', frame_idx) + jpg.tobytes()
        try:
            _send_msg(conn, MSG_FRAME, payload)
        except Exception as e:
            log.warning(f"[OffloadLink] send failed: {e}"); self._drop()

    def get_result(self):
        with self._res_lock: return self._result

    def stop(self):
        self._stop.set(); self._drop()

    def _recv_loop(self):
        with self._conn_lock: conn = self._conn
        try:
            while not self._stop.is_set():
                msg_type, payload = _recv_msg(conn)
                if msg_type == MSG_RESULT:
                    with self._res_lock:
                        self._result = json.loads(payload.decode())
                elif msg_type == MSG_PING:
                    _send_msg(conn, MSG_PONG)
                elif msg_type == MSG_SEED_START:
                    import shutil
                    if PI_SEEDS_DIR.exists():
                        shutil.rmtree(str(PI_SEEDS_DIR))
                    PI_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
                    info = json.loads(payload.decode())
                    log.info(f"[Seeds] wiped — expecting {info['count']} seeds")
                elif msg_type == MSG_SEED_FILE:
                    nl     = payload.index(b'\n')
                    header = json.loads(payload[:nl].decode())
                    raw    = payload[nl+1:]
                    out    = PI_SEEDS_DIR / header['filename']
                    out.write_bytes(raw)
                    log.info(f"[Seeds] {header['filename']} ({len(raw)//1024}KB)")
                elif msg_type == MSG_SEED_DONE:
                    seeds = list(PI_SEEDS_DIR.glob('*'))
                    log.info(f"[Seeds] done — {len(seeds)} seeds in {PI_SEEDS_DIR}")
        except Exception as e:
            if not self._stop.is_set():
                log.warning(f"[OffloadLink] recv ended: {e}")
        finally:
            self._drop()

    def _ping_loop(self):
        while not self._stop.is_set():
            time.sleep(OFFLOAD_PING_INTERVAL)
            with self._conn_lock: conn = self._conn
            if conn is None: break
            try: _send_msg(conn, MSG_PING)
            except Exception: self._drop(); break

    def _drop(self):
        with self._conn_lock:
            if self._conn:
                try: self._conn.close()
                except Exception: pass
                self._conn = None


# ── offload draw ──────────────────────────────────────────────────────────────

def _offload_draw(bgr, result, frame_idx, lat_ms):
    disp      = bgr.copy()
    state     = result.get('state', '?')
    confirmed = result.get('confirmed', False)
    lock_box  = result.get('lock_box')
    col       = (0, 255, 0) if confirmed else (0, 0, 255)
    if lock_box:
        x1, y1, x2, y2 = lock_box
        cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
        cv2.putText(disp, state, (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(disp,
        f"OFFLOAD f:{frame_idx} lat:{lat_ms:.0f}ms {state} "
        f"score:{result.get('best_score',0):.3f} M:{result.get('orb_matches',0)}",
        (8, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1)
    return disp


# ── MODE: offload ─────────────────────────────────────────────────────────────

def run_offload(cam_index=1, tcp_port=OFFLOAD_DEFAULT_PORT, display_port=8081,
                width=640, height=480, jpeg_quality=75):
    run_id = str(uuid.uuid4())[:8].upper()
    log.info("=" * 60)
    log.info(f"  Mode   : offload  Run ID : {run_id}")
    log.info(f"  Camera : picam/{cam_index}  {width}x{height}")
    log.info("=" * 60)

    pc_ip = _discover_pc()
    cap   = PiCamCapture(cam_index, width=width, height=height)
    link  = OffloadLink(pc_ip=pc_ip, tcp_port=tcp_port)
    web   = WebStreamer(port=display_port)

    frame_idx   = 0
    last_result = None
    push_ts     = {}

    try:
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                time.sleep(0.02); continue
            frame_idx += 1

            link.send_frame(bgr, frame_idx, jpeg_quality)
            push_ts[frame_idx] = time.perf_counter()
            if len(push_ts) > 120:
                del push_ts[min(push_ts)]

            rx = link.get_result()
            if rx is not None and rx is not last_result:
                fi  = rx.get('frame_idx', frame_idx)
                rx['_lat'] = (time.perf_counter() - push_ts.get(fi, time.perf_counter())) * 1000
                last_result = rx
                # Share result with autopilot.py
                try:
                    import json as _json
                    open('/tmp/orb_result.json', 'w').write(_json.dumps(rx))
                except Exception:
                    pass
                # Save bounding box snapshot for centering.py
                try:
                    box = rx.get('lock_box')
                    if rx.get('confirmed') and box:
                        x1,y1,x2,y2 = box
                        x1=max(0,x1); y1=max(0,y1)
                        x2=min(bgr.shape[1],x2); y2=min(bgr.shape[0],y2)
                        if x2>x1 and y2>y1:
                            snap = bgr[y1:y2, x1:x2]
                            cv2.imwrite('/tmp/orb_snapshot.jpg', snap)
                except Exception:
                    pass

            if last_result:
                annotated = _offload_draw(bgr, last_result, frame_idx, last_result.get('_lat', 0))
            else:
                annotated = bgr.copy()
                cv2.putText(annotated, "waiting for PC ...",
                            (8, annotated.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

            web.show(annotated)
            if web.waitKey() == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); link.stop(); web.stop()
        log.info(f"  offload done: {frame_idx} frames")


# ── MODE: bench-live ──────────────────────────────────────────────────────────

def run_bench_live(config, cam_index=0, source='picam'):
    import yaml
    from orb_core import ORBLocker, SeedManager
    with open(config) as f:
        cfg = yaml.safe_load(f)
    paths    = cfg.get('paths', {})
    ref_path = paths.get('ref_path', '')
    out_dir  = paths.get('output_folder', 'output')
    os.makedirs(out_dir, exist_ok=True)

    seed_dir = paths.get('seed_dir')
    if seed_dir:
        exts  = {'.jpg','.jpeg','.png','.bmp'}
        seeds = sorted(str(p) for p in Path(seed_dir).iterdir() if p.suffix.lower() in exts)
    else:
        seeds = [ref_path]

    locker = ORBLocker(cfg, seed_manager=SeedManager(seeds))

    if source == 'picam':
        cap = PiCamCapture(cam_index)
    else:
        import rclpy
        from orb_core import TrackState, MissionState
        # ROS path unchanged from original
        raise NotImplementedError("use --source picam for bench-live on Pi")

    web       = WebStreamer()
    frame_idx = 0
    try:
        while True:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                time.sleep(0.02); continue
            frame_idx += 1
            gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            result = locker.process_frame(gray)
            disp   = locker.draw_annotated(bgr, result)
            cv2.putText(disp, f"BENCH-LIVE f:{frame_idx} {result['state'].name}",
                        (8, disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            web.show(disp)
            if web.waitKey() == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release(); web.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="ORB Pi entry point")
    p.add_argument('--mode',    choices=['bench-live','offload'], default='offload')
    p.add_argument('--cam',     type=int, default=1)
    p.add_argument('--config',  default='orb.yaml', help='bench-live only')
    p.add_argument('--source',  default='picam',    help='bench-live only')
    p.add_argument('--port',    type=int, default=OFFLOAD_DEFAULT_PORT)
    p.add_argument('--width',   type=int, default=640)
    p.add_argument('--height',  type=int, default=480)
    p.add_argument('--quality', type=int, default=75)
    return p.parse_args()

def main():
    args = _parse()
    if args.mode == 'offload':
        run_offload(cam_index=args.cam, tcp_port=args.port,
                    width=args.width, height=args.height,
                    jpeg_quality=args.quality)
    elif args.mode == 'bench-live':
        run_bench_live(config=args.config, cam_index=args.cam, source=args.source)

if __name__ == '__main__':
    main()
