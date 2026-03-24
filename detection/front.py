#!/usr/bin/env python3
"""
MISSILE LOCK MODE – LIVE CAMERA (FIXED FOR REAL TARGETS)
- Stricter validation to avoid false positives
- Motion tracking to maintain lock on real target
- Background subtraction to ignore static patterns
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque

# Suppress Qt font warnings
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false;qt.qpa.fonts=false"

# ==================== CONFIG ====================
REF_PATH              = "/home/daniel/Desktop/drone/seed/3d.png"
CAMERA_ID             = 0
SESSION_DURATION      = 180  # 3 minutes

# MUCH STRICTER THRESHOLDS for live camera
TEMPLATE_THRESHOLD    = 0.6    # Increased from 0.3 - much stricter
ORB_CONFIRM_THRESHOLD = 15.0   # Increased from 6.0 - need more matches
ORB_RATIO_TEST        = 0.7    # Stricter ratio test

# Target validation
MIN_MATCHES_FOR_LOCK  = 20     # Minimum ORB matches to consider it real
MAX_MOVEMENT_PER_FRAME = 100    # Max pixels target can move between frames
HISTORY_SIZE          = 10      # Track last N positions for smoothing

DISPLAY_SCALE         = 0.65
REF_DISPLAY_WIDTH     = 110
LOCK_COLOR            = (0, 0, 255)   # Red
SEARCH_COLOR          = (0, 255, 0)   # Green
CONFIRMED_COLOR       = (0, 255, 255) # Yellow for confirmed targets
OUTPUT_FOLDER         = "live_screenshots"

ZOOM_PAD              = 25
FRAME_SKIP            = 1  # Process every frame for better tracking

# ==================== TARGET TRACKER CLASS ====================
class TargetTracker:
    def __init__(self, max_history=HISTORY_SIZE):
        self.positions = deque(maxlen=max_history)
        self.scores = deque(maxlen=max_history)
        self.current_target = None
        self.lock_confidence = 0
        self.consecutive_locks = 0
        self.last_valid_position = None
        
    def update(self, position, score, orb_matches):
        """Update tracker with new detection"""
        is_valid = False
        
        # Check if detection is plausible
        if position and score > TEMPLATE_THRESHOLD and orb_matches > ORB_CONFIRM_THRESHOLD:
            
            # Check movement continuity (if we have previous position)
            if self.last_valid_position is not None:
                movement = np.sqrt((position[0] - self.last_valid_position[0])**2 + 
                                 (position[1] - self.last_valid_position[1])**2)
                
                # If movement is too large, it's probably a false positive
                if movement > MAX_MOVEMENT_PER_FRAME:
                    print(f"Rejected: Movement too large ({movement:.1f}px)")
                    self.consecutive_locks = max(0, self.consecutive_locks - 1)
                    return False
            
            # If we have enough ORB matches, it's probably real
            if orb_matches >= MIN_MATCHES_FOR_LOCK:
                is_valid = True
                self.consecutive_locks += 1
            else:
                # Borderline case - need multiple consistent detections
                self.consecutive_locks = max(0, self.consecutive_locks - 0.5)
                
            if is_valid:
                self.positions.append(position)
                self.scores.append(score)
                self.last_valid_position = position
                
                # Calculate smoothed position
                if len(self.positions) > 2:
                    self.current_target = np.mean(self.positions, axis=0).astype(int)
                else:
                    self.current_target = position
                    
                self.lock_confidence = min(1.0, self.consecutive_locks / 5.0)
                return True
        
        # No valid detection
        self.consecutive_locks = max(0, self.consecutive_locks - 0.2)
        self.lock_confidence = max(0, self.lock_confidence - 0.1)
        return False
    
    def get_tracked_target(self):
        """Get current tracked position"""
        if self.lock_confidence > 0.3 and self.current_target is not None:
            return self.current_target
        return None


# ==================== BACKGROUND SUBTRACTOR ====================
class BackgroundFilter:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.foreground_mask = None
        
    def filter_frame(self, frame_gray):
        """Remove static background to focus on moving targets"""
        self.foreground_mask = self.backSub.apply(frame_gray)
        
        # Apply some morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        self.foreground_mask = cv2.morphologyEx(self.foreground_mask, cv2.MORPH_OPEN, kernel)
        self.foreground_mask = cv2.morphologyEx(self.foreground_mask, cv2.MORPH_CLOSE, kernel)
        
        return self.foreground_mask


# ==================== MAIN ====================
def main():
    print("=== MISSILE LOCK MODE – LIVE CAMERA (STRICT TARGET VALIDATION) ===\n")
    print(f"Session duration: {SESSION_DURATION} seconds")
    print(f"Thresholds: TM>{TEMPLATE_THRESHOLD}, ORB>{ORB_CONFIRM_THRESHOLD}, MIN>{MIN_MATCHES_FOR_LOCK}")
    print("Press 'q' to quit, 'd' to debug view\n")

    # Load reference
    if not Path(REF_PATH).is_file():
        print(f"ERROR: Reference not found → {REF_PATH}")
        return
    
    ref_bgr = cv2.imread(REF_PATH)
    if ref_bgr is None:
        print("ERROR: Could not load reference image")
        return
        
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = ref_gray.shape
    
    # Extract reference features
    orb_ref = cv2.ORB_create(nfeatures=5000)
    kp_ref, des_ref = orb_ref.detectAndCompute(ref_gray, None)
    if des_ref is None or len(des_ref) < 50:
        print("ERROR: Reference image doesn't have enough features!")
        return
    
    print(f"Reference loaded: {ref_w}x{ref_h}, {len(des_ref)} features")

    # Small overlay reference
    ref_small = cv2.resize(ref_bgr, (REF_DISPLAY_WIDTH, int(REF_DISPLAY_WIDTH * ref_bgr.shape[0] / ref_bgr.shape[1])))

    # Multi-scale templates (narrower range for live)
    scales = [0.9, 1.0, 1.1]  # Reduced range to avoid false positives
    scaled_templates = []
    for s in scales:
        try:
            tmpl = cv2.resize(ref_gray, (int(ref_w * s), int(ref_h * s)))
            scaled_templates.append((s, tmpl))
        except:
            continue

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_ID}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for stability
    
    print(f"Camera opened: {CAMERA_ID}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # Prepare output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Initialize trackers
    target_tracker = TargetTracker()
    background_filter = BackgroundFilter()
    
    # Statistics
    frame_counter = 0
    lock_count = 0
    false_positives = 0
    prev_time = time.time()
    start_time = time.time()
    
    # ORB detector for live frames
    orb_live = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    debug_mode = False

    # ==================== MAIN LOOP ====================
    while time.time() - start_time < SESSION_DURATION:
        # Calculate remaining time
        remaining = SESSION_DURATION - (time.time() - start_time)
        mins, secs = divmod(int(remaining), 60)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get foreground mask (moving objects only)
        fg_mask = background_filter.filter_frame(gray)
        
        # Apply mask to focus on moving areas (optional)
        # masked_gray = cv2.bitwise_and(gray, gray, mask=fg_mask)
        
        # ==================== TEMPLATE MATCHING ====================
        best_score = 0
        best_pt = None
        best_scale = 1.0
        
        for scale, tmpl in scaled_templates:
            # Skip if template too large
            if tmpl.shape[0] > gray.shape[0] or tmpl.shape[1] > gray.shape[1]:
                continue
                
            # Try matching on original and masked version
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_pt = max_loc
                best_scale = scale

        # ==================== ORB VERIFICATION ====================
        good_matches = []
        orb_matches = 0
        roi_rect = None
        
        if best_score > TEMPLATE_THRESHOLD and best_pt is not None:
            x, y = best_pt
            sw = int(ref_w * best_scale)
            sh = int(ref_h * best_scale)
            
            # Extract ROI
            rx1 = max(0, x - ZOOM_PAD)
            ry1 = max(0, y - ZOOM_PAD)
            rx2 = min(gray.shape[1], x + sw + ZOOM_PAD)
            ry2 = min(gray.shape[0], y + sh + ZOOM_PAD)
            roi = gray[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                # Detect features in ROI
                kp_roi, des_roi = orb_live.detectAndCompute(roi, None)
                
                if des_roi is not None and len(des_roi) > 5:
                    # Match with reference
                    matches = bf.knnMatch(des_ref, des_roi, k=2)
                    
                    for pair in matches:
                        if len(pair) == 2:
                            m, n = pair
                            if m.distance < ORB_RATIO_TEST * n.distance:
                                good_matches.append(m)
                    
                    orb_matches = len(good_matches)
                    roi_rect = (rx1, ry1, rx2, ry2)
        
        # ==================== TARGET VALIDATION ====================
        is_locked = False
        is_target_real = False
        
        # Update tracker
        if best_pt is not None:
            was_updated = target_tracker.update(best_pt, best_score, orb_matches)
            tracked_pos = target_tracker.get_tracked_target()
            
            # Target is real if we have enough ORB matches
            is_target_real = orb_matches >= MIN_MATCHES_FOR_LOCK
            is_locked = target_tracker.lock_confidence > 0.5 and tracked_pos is not None
        
        # ==================== DISPLAY ====================
        display = frame.copy()
        
        # FPS calculation
        t_now = time.time()
        fps = 1.0 / (t_now - prev_time) if t_now > prev_time else 0.0
        prev_time = t_now
        
        # Draw detection results
        if roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            
            # Different colors based on confidence
            if is_target_real:
                color = CONFIRMED_COLOR  # Yellow for confirmed real target
                cv2.putText(display, f"REAL TARGET! ({orb_matches} matches)", 
                           (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Save screenshot when we get a real lock
                if lock_count == 0 or frame_counter % 50 == 0:
                    timestamp = time.strftime("%H%M%S")
                    fname = f"{OUTPUT_FOLDER}/lock_{timestamp}_frame{frame_counter:04d}.png"
                    cv2.imwrite(fname, display)
                    print(f"\n✅ REAL TARGET LOCKED! {orb_matches} matches - Screenshot saved")
                    lock_count += 1
            elif orb_matches > ORB_CONFIRM_THRESHOLD:
                color = LOCK_COLOR  # Red for possible target
            else:
                color = SEARCH_COLOR  # Green for false positive
            
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 3)
            
            # Add match count
            cv2.putText(display, f"ORB: {orb_matches}", (rx1, ry2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Draw tracked position
        tracked_pos = target_tracker.get_tracked_target()
        if tracked_pos is not None:
            cv2.circle(display, tuple(tracked_pos), 10, CONFIRMED_COLOR, 2)
            cv2.circle(display, tuple(tracked_pos), 3, CONFIRMED_COLOR, -1)
        
        # ==================== OVERLAY INFO ====================
        # Time and FPS
        cv2.putText(display, f"Time: {mins:02d}:{secs:02d} | FPS: {fps:.1f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Detection info
        tm_score_text = f"TM: {best_score:.2f}"
        orb_text = f"ORB: {orb_matches}/{MIN_MATCHES_FOR_LOCK}"
        conf_text = f"Conf: {target_tracker.lock_confidence:.2f}"
        cv2.putText(display, f"{tm_score_text} | {orb_text} | {conf_text}",
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Status
        if is_target_real:
            status_text = "🔴 REAL TARGET LOCKED!"
            status_col = CONFIRMED_COLOR
        elif is_locked:
            status_text = "🟡 TRACKING (low confidence)"
            status_col = LOCK_COLOR
        else:
            status_text = "🟢 SEARCHING"
            status_col = SEARCH_COLOR
            
        cv2.putText(display, status_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)
        
        # False positive counter
        if not is_target_real and orb_matches > ORB_CONFIRM_THRESHOLD:
            false_positives += 1
        cv2.putText(display, f"False +: {false_positives}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        # Reference overlay
        rh, rw = ref_small.shape[:2]
        xoff = display.shape[1] - rw - 10
        yoff = 10
        display[yoff:yoff + rh, xoff:xoff + rw] = ref_small
        
        # Debug view
        if debug_mode:
            # Show foreground mask
            fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            small_fg = cv2.resize(fg_colored, (320, 240))
            display[10:250, 10:330] = small_fg
        
        # ==================== SHOW ====================
        small_display = cv2.resize(display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("MISSILE LOCK - REAL TARGETS ONLY", small_display)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        if key == ord(' '):
            cv2.waitKey(0)
        
        # Progress update
        if frame_counter % 30 == 0:
            print(f"\rProgress: {mins:02d}:{secs:02d} | Locks: {lock_count} | FP: {false_positives}", end="")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Session summary
    print(f"\n\n{'='*50}")
    print(f"SESSION COMPLETE")
    print(f"Duration: {SESSION_DURATION} seconds")
    print(f"Frames processed: {frame_counter}")
    print(f"✅ REAL TARGET LOCKS: {lock_count}")
    print(f"❌ False positives filtered: {false_positives}")
    print(f"Screenshots saved in: {OUTPUT_FOLDER}")
    print(f"{'='*50}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user")
        cv2.destroyAllWindows()