# Optimized Helmet Detection for Raspberry Pi 5 with AI Hat
import numpy as np
import cv2
import torch
import time
import os
import threading
from queue import Queue
from collections import deque
import json

# Check for Hailo/Edge TPU availability
try:
    import hailo  # For Hailo AI accelerator
    HAS_HAILO = True
except ImportError:
    HAS_HAILO = False
    print("[WARN] Hailo not found, falling back to CPU")

# Import LightGlue components
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

class PiHelmetTracker:
    def __init__(self, seed_path, camera_id=0, 
                 process_every_n_frames=2,
                 match_threshold=0.25,
                 min_matches=4,
                 use_npu=True):
        
        self.seed_path = seed_path
        self.camera_id = camera_id
        self.process_every_n_frames = process_every_n_frames
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        self.use_npu = use_npu and HAS_HAILO
        
        # Check seed image
        if not os.path.exists(seed_path):
            raise FileNotFoundError(f"Seed image not found: {seed_path}")
        
        # Setup device - CPU only for now (NPU integration would require model conversion)
        self.device = torch.device('cpu')
        print(f"[INFO] Using device: {self.device}")
        
        # Performance optimizations for Pi
        self.max_keypoints = 512  # Reduced for speed
        self.image_resolution = (640, 480)  # Lower resolution for better FPS
        
        # Initialize models with optimized settings
        self._init_models()
        
        # Load and process seed image
        self._load_seed_image()
        
        # Tracking variables
        self.last_box = None
        self.frame_count = 0
        self.fps = 0
        self.fps_samples = deque(maxlen=30)
        self.last_time = time.time()
        
        # Threading for smoother display
        self.display_queue = Queue(maxsize=2)
        self.running = False
        
        # Performance metrics
        self.processing_times = deque(maxlen=30)
        self.match_counts = []
        
    def _init_models(self):
        """Initialize models with Pi-optimized settings"""
        print("[INFO] Initializing models...")
        
        # Use smaller model or quantized version if available
        self.extractor = SuperPoint(
            max_num_keypoints=self.max_keypoints,
            detection_threshold=0.0005  # Lower threshold for more keypoints
        ).eval().to(self.device)
        
        self.matcher = LightGlue(
            features='superpoint',
            depth_confidence=0.9,  # Faster matching
            width_confidence=0.9
        ).eval().to(self.device)
        
        print("[INFO] Models initialized successfully")
    
    def _load_seed_image(self):
        """Load and preprocess seed image"""
        print(f"[INFO] Loading seed image: {self.seed_path}")
        
        # Load and resize seed image to match processing resolution
        image0 = load_image(self.seed_path).to(self.device)
        
        # Resize if needed
        if image0.shape[2] > self.image_resolution[0]:
            image0 = torch.nn.functional.interpolate(
                image0.unsqueeze(0), 
                size=(self.image_resolution[1], self.image_resolution[0]),
                mode='bilinear'
            ).squeeze(0)
        
        with torch.no_grad():
            self.feats0 = self.extractor.extract(image0)
        
        print(f"[INFO] Seed image processed: {self.feats0['keypoints'].shape[0]} keypoints")
    
    def _preprocess_frame(self, frame):
        """Optimized frame preprocessing for Pi"""
        # Resize for faster processing
        if frame.shape[1] != self.image_resolution[0]:
            frame = cv2.resize(frame, self.image_resolution, interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        
        return frame_tensor.to(self.device)
    
    def _process_frame(self, frame):
        """Process single frame for object tracking"""
        start_time = time.time()
        
        frame_tensor = self._preprocess_frame(frame)
        
        with torch.no_grad():
            feats1 = self.extractor.extract(frame_tensor)
            matches01 = self.matcher({'image0': self.feats0, 'image1': feats1})
        
        # Remove batch dimension
        feats1_rbd, matches01_rbd = rbd(feats1), rbd(matches01)
        scores = matches01_rbd['scores']
        indices = matches01_rbd['matches'][scores > self.match_threshold]
        
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        if len(indices) < self.min_matches:
            return None
        
        # Get matched keypoints
        kpts1 = feats1_rbd['keypoints'][indices[..., 1]].cpu().numpy()
        
        # Adaptive clustering based on match count
        if len(kpts1) > 10:
            # Use robust median filtering
            median_pt = np.median(kpts1, axis=0)
            dist = np.linalg.norm(kpts1 - median_pt, axis=1)
            percentile = min(85, 100 - (100 // len(kpts1)))
            cluster_pts = kpts1[dist < np.percentile(dist, percentile)]
        else:
            cluster_pts = kpts1
        
        if len(cluster_pts) < 4:
            return None
        
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(np.float32(cluster_pts).reshape(-1, 1, 2))
        
        # Add adaptive padding
        padding = max(15, min(w, h) // 5)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + (padding * 2))
        h = min(frame.shape[0] - y, h + (padding * 2))
        
        self.match_counts.append(len(indices))
        
        return (x, y, w, h, len(indices))
    
    def _update_fps(self):
        """Calculate moving average FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            self.fps_samples.append(1.0 / elapsed)
            self.fps = np.mean(self.fps_samples)
        self.last_time = current_time
    
    def draw_enhanced_ui(self, frame, box=None):
        """Draw UI with performance metrics"""
        display = frame.copy()
        
        # Draw bounding box
        if box:
            x, y, w, h, pts_count = box
            
            # Dynamic color based on confidence
            confidence = min(1.0, pts_count / 20.0)
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            
            # Draw bounding box with rounded corners
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
            
            # Draw match count badge
            badge_text = f"{pts_count}"
            badge_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display, (x + 5, y - 25), (x + badge_size[0] + 10, y - 5), color, -1)
            cv2.putText(display, badge_text, (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Performance metrics panel (semi-transparent)
        overlay = display.copy()
        panel_y = 10
        panel_h = 110
        
        # Dark semi-transparent panel
        cv2.rectangle(overlay, (5, panel_y), (250, panel_y + panel_h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
        
        # Draw metrics
        y_offset = panel_y + 25
        cv2.putText(display, f"FPS: {self.fps:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            cv2.putText(display, f"Proc: {avg_time:.1f}ms", (10, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.match_counts:
            avg_matches = np.mean(self.match_counts[-30:])
            cv2.putText(display, f"Matches: {avg_matches:.0f}", (10, y_offset + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        # Instructions
        instructions = "Q:Quit | S:Save | R:Reset | F:Fullscreen"
        cv2.putText(display, instructions, (10, display.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    def run(self):
        """Main tracking loop with optimizations for Pi"""
        # Initialize camera with lower resolution for better performance
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_resolution[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        print("[INFO] Camera initialized")
        print(f"[INFO] Resolution: {self.image_resolution}")
        
        # Setup window
        window_name = "Helmet Tracker - Raspberry Pi 5"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.image_resolution[0], self.image_resolution[1])
        
        fullscreen = False
        self.running = True
        
        # Frame skipping counter
        process_counter = 0
        
        print("[INFO] Starting tracking loop...")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Failed to grab frame")
                    continue
                
                # Update FPS counter
                self._update_fps()
                
                # Process every Nth frame
                process_counter += 1
                if process_counter >= self.process_every_n_frames:
                    process_counter = 0
                    
                    try:
                        box = self._process_frame(frame)
                        if box:
                            self.last_box = box
                        else:
                            # Use last known box if tracking lost briefly
                            if self.last_box and (self.frame_count - self.last_box_time < 10):
                                pass  # Keep last box
                            else:
                                self.last_box = None
                        self.last_box_time = self.frame_count
                    except Exception as e:
                        print(f"[ERROR] Processing failed: {e}")
                        self.last_box = None
                
                # Draw UI
                display_frame = self.draw_enhanced_ui(frame, self.last_box)
                
                # Display
                cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and self.last_box:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Frame saved: {filename}")
                elif key == ord('r'):
                    # Reset tracker by reloading seed
                    self._load_seed_image()
                    self.last_box = None
                    print("[INFO] Tracker reset")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                            cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, self.image_resolution[0], 
                                       self.image_resolution[1])
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_performance_stats()
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        print(f"Average FPS: {self.fps:.2f}")
        if self.processing_times:
            print(f"Avg processing time: {np.mean(self.processing_times):.2f}ms")
            print(f"Min processing time: {np.min(self.processing_times):.2f}ms")
            print(f"Max processing time: {np.max(self.processing_times):.2f}ms")
        if self.match_counts:
            print(f"Avg matches: {np.mean(self.match_counts):.1f}")
        print(f"Total frames: {self.frame_count}")
        print("="*50)

def main():
    """Main entry point with configuration"""
    # Configuration (adjust based on your needs)
    config = {
        'seed_path': 'benchmark/printer.png',  # Change to your helmet image
        'camera_id': 0,  # Usually 0 for built-in camera
        'process_every_n_frames': 2,  # Process every 2nd frame for ~15 FPS
        'match_threshold': 0.25,  # Lower = more matches, but more false positives
        'min_matches': 4,  # Minimum matches needed for detection
        'use_npu': True  # Use NPU if available
    }
    
    print("""
    ╔══════════════════════════════════════════════════╗
    ║     Raspberry Pi 5 Helmet Detection System      ║
    ║          Powered by LightGlue + AI Hat          ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    try:
        tracker = PiHelmetTracker(**config)
        tracker.run()
    except Exception as e:
        print(f"[ERROR] Failed to start tracker: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())