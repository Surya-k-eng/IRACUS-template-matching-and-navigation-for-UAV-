#!/usr/bin/env python3
"""
MISSILE LOCK MODE
- Template matching + ORB verification + NanoTrack tracking
- Processes ROS bag images from a specified frame range
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

# Suppress Qt font warnings (harmless but noisy)
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false;qt.qpa.fonts=false"

# ==================== CONFIG ====================
REF_PATH              = "/home/daniel/Desktop/drone/seed/seed.png"
BAG_PATH              = "/home/daniel/Downloads/flight_dataset2.bag"
IMAGE_TOPIC           = "/camera/image_raw"

START_FRAME           = 1900
END_FRAME             = 2400          # None = process until end

TEMPLATE_THRESHOLD    = 0.3
ORB_CONFIRM_THRESHOLD = 6.0           # adjust 5.0–8.0 based on false positives
ORB_RATIO_TEST        = 0.75

DISPLAY_SCALE         = 0.65
REF_DISPLAY_WIDTH     = 110
LOCK_COLOR            = (0, 0, 255)   # BGR red
SEARCH_COLOR          = (0, 255, 0)   # BGR green
OUTPUT_FOLDER         = "output_screenshots"

BACKBONE_PATH         = "nanotrack_backbone_sim.onnx"
NECKHEAD_PATH         = "nanotrack_head_sim.onnx"


def main():
    print("=== MISSILE LOCK MODE – TEMPLATE + ORB + NANOTRACK ===\n")
    print(f"Reference image: {REF_PATH}")
    print(f"Bag file:        {BAG_PATH}")
    print(f"Frame range:     {START_FRAME} → {END_FRAME if END_FRAME else 'end of bag'}")
    print()

    # ── Load reference ───────────────────────────────────────────────
    if not Path(REF_PATH).is_file():
        print(f"ERROR: Reference not found → {REF_PATH}")
        return

    ref_bgr  = cv2.imread(REF_PATH)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    if des_ref is None or len(des_ref) == 0:
        print("ERROR: Reference has no ORB features")
        return

    # Small overlay version
    ref_h, ref_w = ref_bgr.shape[:2]
    ref_small = cv2.resize(ref_bgr, (REF_DISPLAY_WIDTH, int(REF_DISPLAY_WIDTH * ref_h / ref_w)))

    # Multi-scale gray templates
    ref_h, ref_w = ref_gray.shape
    scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    scaled_templates = [
        (s, cv2.resize(ref_gray, (int(ref_w * s), int(ref_h * s))))
        for s in scales
    ]

    # ── Read bag images ──────────────────────────────────────────────
    messages = []
    bag_path = Path(BAG_PATH)

    print("Opening ROS bag...")
    with AnyReader([bag_path]) as reader:
        print("Available topics:")
        image_conns = []
        for conn in reader.connections:
            print(f"  {conn.topic:50} {conn.msgtype}")
            if 'image' in conn.topic.lower() or 'Image' in conn.msgtype:
                image_conns.append(conn)

        if not image_conns:
            print("No image topics found.")
            return

        topic = IMAGE_TOPIC
        if topic not in {c.topic for c in image_conns}:
            print(f"\nWARNING: Requested topic '{topic}' not found.")
            print("Available image topics:")
            for i, c in enumerate(image_conns, 1):
                print(f"  {i}) {c.topic}")
            choice = int(input("Select number: ")) - 1
            topic = image_conns[choice].topic

        print(f"\nReading from: {topic}")

        conns = [c for c in reader.connections if c.topic == topic]
        for conn, ts_ns, raw in reader.messages(connections=conns):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                img = message_to_cvimage(msg, 'bgr8')
                messages.append((ts_ns, img))
            except Exception as e:
                print(f"Decode error at {ts_ns}: {e}")

    print(f"Loaded {len(messages)} images\n")
    if not messages:
        return

    # ── Prepare output ───────────────────────────────────────────────
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for f in os.listdir(OUTPUT_FOLDER):
        if f.endswith('.png'):
            os.remove(os.path.join(OUTPUT_FOLDER, f))

    cv2.namedWindow("TEMPLATE + NANOTRACK LOCK", cv2.WINDOW_NORMAL)

    # State
    tracker       = None
    is_locked     = False
    frame_counter = 0
    lock_count    = 0
    lock_records  = []  # (frame, matches, tm_conf, x1,y1,x2,y2)

    print("Playback started.  q = quit, space = pause\n")
    prev_t = time.time()

    for ts_ns, frame in messages:
        frame_counter += 1

        if frame_counter < START_FRAME:
            continue
        if END_FRAME is not None and frame_counter > END_FRAME:
            print(f"Reached end frame {END_FRAME} → stopping")
            break

        if frame is None:
            continue

        t_now = time.time()
        fps = 1 / (t_now - prev_t) if t_now > prev_t else 0
        prev_t = t_now

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ros_sec = ts_ns / 1e9

        cv2.putText(display, f"Frame {frame_counter}/{len(messages)}  FPS:{fps:.1f}  ROS:{ros_sec:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ── Update existing tracker ──────────────────────────────────
        if tracker is not None:
            ok, bbox = tracker.update(frame)
            if ok:
                is_locked = True
                x, y, w, h = map(int, bbox)
                cv2.rectangle(display, (x, y), (x + w, y + h), LOCK_COLOR, 3)
                cv2.putText(display, "LOCKED (NanoTrack)", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, LOCK_COLOR, 3)
            else:
                print(f"Frame {frame_counter}: NanoTrack LOST → re-init needed")
                tracker = None
                is_locked = False

        # ── Detect & verify if no tracker ────────────────────────────
        if tracker is None:
            best_score = 0
            best_pt    = None
            best_scale = 1.0

            for scale, tmpl in scaled_templates:
                res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_score:
                    best_score = max_val
                    best_pt    = max_loc
                    best_scale = scale

            cv2.putText(display, f"TM Conf: {best_score:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            if best_score > TEMPLATE_THRESHOLD and best_pt is not None:
                x, y = best_pt
                sw = int(ref_w * best_scale)
                sh = int(ref_h * best_scale)

                cv2.rectangle(display, (x, y), (x + sw, y + sh), SEARCH_COLOR, 2)

                # ROI + padding for ORB
                pad = 45
                rx1 = max(0, x - pad)
                ry1 = max(0, y - pad)
                rx2 = min(frame.shape[1], x + sw + pad)
                ry2 = min(frame.shape[0], y + sh + pad)

                roi = gray[ry1:ry2, rx1:rx2]
                if roi.size == 0:
                    continue

                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255,0,255), 2)

                kp_roi, des_roi = orb.detectAndCompute(roi, None)
                if des_roi is not None and len(des_roi) > 0:
                    matches = bf.knnMatch(des_ref, des_roi, k=2)

                    good_matches = []
                    for pair in matches:
                        if len(pair) == 2:
                            m, n = pair
                            if m.distance < ORB_RATIO_TEST * n.distance:
                                good_matches.append(m)

                    good_matches = sorted(good_matches, key=lambda m: m.distance)[:30]

                    cv2.putText(display, f"ORB: {len(good_matches)}", (rx1, ry1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

                    if len(good_matches) >= ORB_CONFIRM_THRESHOLD:
                        lock_count += 1
                        print(f"Frame {frame_counter:4d} | TM:{best_score:.3f} | ORB:{len(good_matches):2d} → LOCK")
                        lock_records.append((frame_counter, len(good_matches), best_score, rx1, ry1, rx2, ry2))

                        # ── Init NanoTrack ───────────────────────────────
                        try:
                            ix, iy = int(x), int(y)
                            iw, ih = int(sw), int(sh)

                            if iw <= 0 or ih <= 0 or ix < 0 or iy < 0:
                                print(f"Invalid rect ({ix},{iy},{iw},{ih}) → skip init")
                                tracker = None
                                continue

                            print(f"Init NanoTrack → rect: ({ix},{iy},{iw},{ih})")

                            params = cv2.TrackerNano_Params()
                            params.backbone = BACKBONE_PATH
                            params.neckhead = NECKHEAD_PATH
                            tracker = cv2.TrackerNano_create(params)

                            if tracker.init(frame, (ix, iy, iw, ih)):
                                is_locked = True
                                print(" → NanoTrack initialized OK")
                            else:
                                print(" → tracker.init() returned False")
                                tracker = None

                        except Exception as e:
                            print(f"NanoTrack init failed: {e}")
                            tracker = None

        # ── UI overlays ──────────────────────────────────────────────
        rh, rw = ref_small.shape[:2]
        xoff = display.shape[1] - rw - 10
        yoff = 10
        display[yoff:yoff + rh, xoff:xoff + rw] = ref_small

        status_text = "LOCKED" if is_locked else "SEARCHING"
        status_col  = LOCK_COLOR if is_locked else SEARCH_COLOR
        cv2.putText(display, f"Status: {status_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

        # ── Show frame ───────────────────────────────────────────────
        small_display = cv2.resize(display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("TEMPLATE + NANOTRACK LOCK", small_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    # ── Save results ─────────────────────────────────────────────────
    if lock_records:
        lock_records.sort(key=lambda r: (-r[2], -r[1]))  # best TM conf, then most ORB matches

        top_n = lock_records[:5]
        for rank, (fnum, nmatch, conf, x1, y1, x2, y2) in enumerate(top_n, 1):
            if fnum - 1 < len(messages):
                _, best_img = messages[fnum - 1]
                crop = best_img[y1:y2, x1:x2]
                fname = f"{OUTPUT_FOLDER}/top{rank}_frame{fnum:04d}_m{nmatch:02d}_c{conf:.3f}.png"
                cv2.imwrite(fname, crop)
                print(f"Saved top {rank}: {fname}")

        with open(f"{OUTPUT_FOLDER}/detections.txt", "w") as f:
            f.write("Frame  Matches  TM_Conf\n")
            for fn, nm, cf, *_ in lock_records:
                f.write(f"{fn:5d}  {nm:7d}  {cf:.3f}\n")

    print(f"\nFinished. Confirmed locks: {lock_count}")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
