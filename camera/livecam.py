#!/usr/bin/env python3
"""
MISSILE LOCK MODE – DYNAMIC LIVE BEST-GUESS (PARALLELIZED)
- Template Matching + ORB verification + NanoTrack
- Parallel frame processing for speed
- Dynamic zoom highlighting with confidence check
- FPS overlay
- OPTIMIZED FOR PI 4 (4 CORES)
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import multiprocessing as mp

# Suppress Qt font warnings
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false;qt.qpa.fonts=false"

# ==================== CONFIG ====================
REF_PATH              = "/home/daniel/Desktop/drone/seed/seed.png"
BAG_PATH              = "/home/daniel/Downloads/flight_dataset2.bag"
IMAGE_TOPIC           = "/camera/image_raw"

START_FRAME           = 1900
END_FRAME             = 2400         # None = process until end

TEMPLATE_THRESHOLD    = 0.3
ORB_CONFIRM_THRESHOLD = 6.0
ORB_RATIO_TEST        = 0.75

DISPLAY_SCALE         = 0.65
REF_DISPLAY_WIDTH     = 110
LOCK_COLOR            = (0, 0, 255)   # Red
SEARCH_COLOR          = (0, 255, 0)   # Green
ZOOM_COLOR            = (0, 0, 255)   # Red for improved zoom
OUTPUT_FOLDER         = "output_screenshots"

BACKBONE_PATH         = "nanotrack_backbone_sim.onnx"
NECKHEAD_PATH         = "nanotrack_head_sim.onnx"

ZOOM_PAD              = 45   # extra pixels around TM for zoom

# PI 4 OPTIMIZATION: Use 2-3 workers max (leave 1-2 cores for display/OS)
# Pi 4 has 4 cores total, using all 4 causes severe lag
MAX_WORKERS = 2  # Changed from 4 to 2 for Pi 4

# PI 4 OPTIMIZATION: Reduce ORB features to save CPU
ORB_FEATURES = 1500  # Reduced from 3000 (saves ~40% CPU on Pi)

# PI 4 OPTIMIZATION: Fewer scales for template matching
SCALES = [0.9, 1.0, 1.1]  # Reduced from 6 scales to 3 (50% less work)


# ==================== FRAME PROCESSING FUNCTION ====================
def process_frame(frame_gray, ref_gray, des_ref, scaled_templates, frame_idx):
    """
    Process a single frame: Template matching + ORB
    Returns frame_idx and results dictionary
    """
    ref_h, ref_w = ref_gray.shape
    best_score = 0
    best_pt = None
    best_scale = 1.0

    for scale, tmpl in scaled_templates:
        res = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_pt = max_loc
            best_scale = scale

    result = {
        'best_score': best_score,
        'best_pt': best_pt,
        'best_scale': best_scale,
        'good_matches_count': 0,
        'roi_rect': None
    }

    # ORB verification
    if best_score > TEMPLATE_THRESHOLD and best_pt is not None:
        x, y = best_pt
        sw = int(ref_w * best_scale)
        sh = int(ref_h * best_scale)
        rx1 = max(0, x - ZOOM_PAD)
        ry1 = max(0, y - ZOOM_PAD)
        rx2 = min(frame_gray.shape[1], x + sw + ZOOM_PAD)
        ry2 = min(frame_gray.shape[0], y + sh + ZOOM_PAD)
        roi = frame_gray[ry1:ry2, rx1:rx2]

        # PI 4 OPTIMIZATION: Use fewer ORB features
        orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        _, des_roi = orb.detectAndCompute(roi, None)
        good_matches = []

        if des_roi is not None and len(des_roi) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des_ref, des_roi, k=2)
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < ORB_RATIO_TEST * n.distance:
                        good_matches.append(m)

        result['good_matches_count'] = len(good_matches)
        result['roi_rect'] = (rx1, ry1, rx2, ry2)

    return frame_idx, result


# ==================== MAIN ====================
def main():
    print("=== MISSILE LOCK MODE – PI 4 OPTIMIZED (2 WORKERS) ===\n")
    print(f"Pi 4 Optimizations:")
    print(f"  - Workers: {MAX_WORKERS}/4 cores")
    print(f"  - ORB Features: {ORB_FEATURES} (was 3000)")
    print(f"  - Template Scales: {len(SCALES)} (was 6)")
    print(f"  - CPU cores available: {mp.cpu_count()}\n")

    # Load reference
    if not Path(REF_PATH).is_file():
        print(f"ERROR: Reference not found → {REF_PATH}")
        return
    ref_bgr  = cv2.imread(REF_PATH)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = ref_gray.shape
    
    # PI 4 OPTIMIZATION: Use fewer ORB features for reference too
    orb_ref = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp_ref, des_ref = orb_ref.detectAndCompute(ref_gray, None)
    if des_ref is None or len(des_ref) == 0:
        print("ERROR: Reference has no ORB features")
        return

    # Small overlay reference
    ref_small = cv2.resize(ref_bgr, (REF_DISPLAY_WIDTH, int(REF_DISPLAY_WIDTH * ref_bgr.shape[0] / ref_bgr.shape[1])))

    # PI 4 OPTIMIZATION: Use fewer scales (3 instead of 6)
    scales = SCALES
    scaled_templates = [(s, cv2.resize(ref_gray, (int(ref_w * s), int(ref_h * s)))) for s in scales]

    # Load ROS bag images
    messages = []
    bag_path = Path(BAG_PATH)
    print("Opening ROS bag...")
    with AnyReader([bag_path]) as reader:
        image_conns = [c for c in reader.connections if 'image' in c.topic.lower() or 'Image' in c.msgtype]
        if not image_conns:
            print("No image topics found.")
            return

        topic = IMAGE_TOPIC if IMAGE_TOPIC in {c.topic for c in image_conns} else image_conns[0].topic
        print(f"Reading from topic: {topic}")
        conns = [c for c in reader.connections if c.topic == topic]

        for conn, ts_ns, raw in reader.messages(connections=conns):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                img = message_to_cvimage(msg, 'bgr8')
                messages.append((ts_ns, img))
            except:
                continue

    if not messages:
        print("No images loaded")
        return
    print(f"Loaded {len(messages)} images")

    # Prepare output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for f in glob.glob(f"{OUTPUT_FOLDER}/*.png"):
        os.remove(f)
    cv2.namedWindow("DYNAMIC MISSILE LOCK", cv2.WINDOW_NORMAL)

    tracker = None
    is_locked = False
    frame_counter = 0
    lock_count = 0
    prev_zoom_conf = 0.0
    prev_time = time.time()

    # ==================== PI 4 OPTIMIZED PARALLEL EXECUTOR ====================
    # PI 4 OPTIMIZATION: Use fewer workers (2 instead of 4)
    # This prevents CPU thrashing and leaves cores for display/OS
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, (ts_ns, frame) in enumerate(messages, 1):
            if idx < START_FRAME or (END_FRAME and idx > END_FRAME):
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            futures[executor.submit(process_frame, gray, ref_gray, des_ref, scaled_templates, idx)] = (idx, frame, ts_ns)

        for future in as_completed(futures):
            frame_idx, frame, ts_ns = futures[future]
            try:
                _, result = future.result()
            except Exception as e:
                print(f"Frame {frame_idx} error: {e}")
                continue

            frame_counter += 1
            display = frame.copy()

            # ==================== FPS ====================
            t_now = time.time()
            fps = 1.0 / (t_now - prev_time) if t_now > prev_time else 0.0
            prev_time = t_now
            ros_sec = ts_ns / 1e9

            cv2.putText(display, f"Frame {frame_idx}/{len(messages)}  FPS:{fps:.1f}  ROS:{ros_sec:.2f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # ==================== DISPLAY RESULTS ====================
            if result['best_score'] > TEMPLATE_THRESHOLD and result['roi_rect'] is not None:
                rx1, ry1, rx2, ry2 = result['roi_rect']
                color = ZOOM_COLOR if result['good_matches_count'] >= ORB_CONFIRM_THRESHOLD else SEARCH_COLOR
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 3)

                # Save screenshot if improved
                if result['good_matches_count'] >= ORB_CONFIRM_THRESHOLD and result['best_score'] > prev_zoom_conf:
                    fname = f"{OUTPUT_FOLDER}/frame{frame_idx:04d}_zoom_improved.png"
                    cv2.imwrite(fname, display)
                    print(f"[Frame {frame_idx}] Zoom improved → screenshot saved: {fname}")
                    prev_zoom_conf = result['best_score']
                    lock_count += 1

            # ==================== Overlay reference ====================
            rh, rw = ref_small.shape[:2]
            xoff = display.shape[1] - rw - 10
            yoff = 10
            display[yoff:yoff + rh, xoff:xoff + rw] = ref_small

            status_text = "LOCKED" if is_locked else "SEARCHING"
            status_col = LOCK_COLOR if is_locked else SEARCH_COLOR
            cv2.putText(display, f"Status: {status_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

            # ==================== SHOW ====================
            small_display = cv2.resize(display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("DYNAMIC MISSILE LOCK", small_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(f"\nFinished. Confirmed locks: {lock_count}")


if __name__ == "__main__":
    # PI 4 OPTIMIZATION: Set process start method to spawn (more stable on Pi)
    mp.set_start_method('spawn', force=True)
    main()