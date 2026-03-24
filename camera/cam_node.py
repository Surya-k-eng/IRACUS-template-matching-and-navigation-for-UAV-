import cv2, rclpy, threading, subprocess, time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CamNode(Node):
    def __init__(self):
        super().__init__('cam_node')
        self.pub = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self.read_loop, daemon=True).start()
        self.create_timer(1.0/30.0, self.callback)

    def fix_exposure(self):
        subprocess.run(['v4l2-ctl','--set-ctrl=auto_exposure=1'], capture_output=True)
        subprocess.run(['v4l2-ctl','--set-ctrl=exposure_time_absolute=300'], capture_output=True)
        subprocess.run(['v4l2-ctl','--set-ctrl=exposure_dynamic_framerate=0'], capture_output=True)

    def read_loop(self):
        last_fix = 0
        while True:
            now = time.time()
            if now - last_fix > 1.0:
                self.fix_exposure()
                last_fix = now
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def callback(self):
        with self.lock:
            frame = self.frame
        if frame is not None:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            msg = self.bridge.cv2_to_imgmsg(grey, 'mono8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)

rclpy.init()
rclpy.spin(CamNode())
