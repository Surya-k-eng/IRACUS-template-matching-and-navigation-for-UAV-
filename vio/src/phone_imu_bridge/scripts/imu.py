#!/usr/bin/env python3
"""
HYPERIMU ROS2 Bridge
Receives IMU data from HYPERIMU Android app via UDP
Publishes to /imu0 topic for VIO/SLAM systems
"""

import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
import threading
import time
import numpy as np
import struct
import json

class HyperIMUBridge(Node):
    def __init__(self):
        super().__init__('hyperimu_bridge')
        
        # Parameters
        self.declare_parameter('listen_port', 2055)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_rate_hz', 100.0)  # 100Hz expected
        self.declare_parameter('gyro_deg_to_rad', True)   # Convert gyro from deg/s to rad/s
        
        self.listen_port = self.get_parameter('listen_port').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = self.get_parameter('publish_rate_hz').value
        self.gyro_deg_to_rad = self.get_parameter('gyro_deg_to_rad').value
        
        # Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu0', 10)
        
        # UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.listen_port))
        self.sock.settimeout(0.1)
        
        # Data buffers
        self.latest_accel = None
        self.latest_gyro = None
        self.latest_gravity = None
        self.latest_orientation = None
        
        # Statistics
        self.packet_count = 0
        self.last_packet_time = time.time()
        self.start_time = time.time()
        
        # Threading
        self.running = True
        self.receive_thread = threading.Thread(target=self.udp_receiver)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        # Timer for publishing (maintain consistent rate)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate, self.publish_imu)
        
        # Status timer
        self.status_timer = self.create_timer(10.0, self.print_status)
        
        self.get_logger().info(f'🚀 HYPERIMU Bridge started')
        self.get_logger().info(f'📡 Listening on UDP port {self.listen_port}')
        self.get_logger().info(f'📤 Publishing to /imu0 at {self.publish_rate} Hz')
        self.get_logger().info(f'🎯 Frame ID: {self.frame_id}')
        
    def udp_receiver(self):
        """Receive and parse UDP packets"""
        buffer = ""
        
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                self.packet_count += 1
                self.last_packet_time = time.time()
                
                # Decode packet
                try:
                    text = data.decode('utf-8', errors='ignore').strip()
                    
                    # HYPERIMU sends comma-separated values
                    # Format: accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,...
                    values = text.split(',')
                    
                    if len(values) >= 6:
                        # Parse accelerometer (m/s²) - these should be linear acceleration
                        accel = (float(values[0]), float(values[1]), float(values[2]))
                        
                        # Parse gyroscope (deg/s or rad/s)
                        gyro = (float(values[3]), float(values[4]), float(values[5]))
                        
                        # Convert gyro from deg/s to rad/s if needed
                        if self.gyro_deg_to_rad and (abs(gyro[0]) > 10 or abs(gyro[1]) > 10 or abs(gyro[2]) > 10):
                            gyro = (gyro[0] * np.pi / 180.0,
                                   gyro[1] * np.pi / 180.0,
                                   gyro[2] * np.pi / 180.0)
                        
                        # Store latest data with timestamp
                        self.latest_accel = (accel, time.time())
                        self.latest_gyro = (gyro, time.time())
                        
                        # If we have extra values, try to parse gravity and orientation
                        if len(values) >= 9:
                            # Some HYPERIMU configurations send gravity or orientation data
                            gravity = (float(values[6]), float(values[7]), float(values[8]))
                            self.latest_gravity = (gravity, time.time())
                            
                        if len(values) >= 12:
                            # Orientation quaternion (w,x,y,z) or Euler angles
                            orientation = (float(values[9]), float(values[10]), float(values[11]), float(values[12]) 
                                         if len(values) > 12 else 1.0)
                            self.latest_orientation = (orientation, time.time())
                            
                except UnicodeDecodeError:
                    # Handle binary data if needed
                    self.handle_binary_data(data)
                except (ValueError, IndexError) as e:
                    self.get_logger().debug(f'Parse error: {e}')
                    
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f'UDP error: {e}')
                time.sleep(0.1)
    
    def handle_binary_data(self, data):
        """Handle binary data format if sent"""
        try:
            # Check if it's 6 floats (24 bytes)
            if len(data) >= 24:
                floats = struct.unpack('<ffffff', data[:24])
                self.latest_accel = ((floats[0], floats[1], floats[2]), time.time())
                self.latest_gyro = ((floats[3], floats[4], floats[5]), time.time())
        except struct.error:
            pass
    
    def publish_imu(self):
        """Publish IMU data at fixed rate"""
        if self.latest_accel is None or self.latest_gyro is None:
            return
        
        # Check if data is stale (older than 0.5 seconds)
        current_time = time.time()
        if current_time - self.latest_accel[1] > 0.5:
            return
        
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        
        # Set linear acceleration (m/s²)
        msg.linear_acceleration = Vector3(
            x=self.latest_accel[0][0],
            y=self.latest_accel[0][1],
            z=self.latest_accel[0][2]
        )
        
        # Set angular velocity (rad/s)
        msg.angular_velocity = Vector3(
            x=self.latest_gyro[0][0],
            y=self.latest_gyro[0][1],
            z=self.latest_gyro[0][2]
        )
        
        # Set orientation if available
        if self.latest_orientation:
            msg.orientation = Quaternion(
                x=self.latest_orientation[0][0],
                y=self.latest_orientation[0][1],
                z=self.latest_orientation[0][2],
                w=self.latest_orientation[0][3] if len(self.latest_orientation[0]) > 3 else 1.0
            )
            msg.orientation_covariance[0] = 0.01  # Small covariance if we have orientation
        else:
            # No orientation data available
            msg.orientation_covariance[0] = -1.0
        
        # Set covariance matrices
        # These values should be tuned based on your phone's sensor quality
        accel_cov = 0.01   # 0.01 m/s²
        gyro_cov = 0.001   # 0.001 rad/s
        
        msg.linear_acceleration_covariance = [
            accel_cov, 0.0, 0.0,
            0.0, accel_cov, 0.0,
            0.0, 0.0, accel_cov
        ]
        
        msg.angular_velocity_covariance = [
            gyro_cov, 0.0, 0.0,
            0.0, gyro_cov, 0.0,
            0.0, 0.0, gyro_cov
        ]
        
        self.imu_pub.publish(msg)
    
    def print_status(self):
        """Print status information periodically"""
        runtime = time.time() - self.start_time
        packets_per_sec = self.packet_count / runtime if runtime > 0 else 0
        
        self.get_logger().info(
            f'📊 Status: {self.packet_count} packets received '
            f'({packets_per_sec:.1f} pps) | '
            f'Last packet: {time.time() - self.last_packet_time:.2f}s ago'
        )
        
        if self.latest_accel:
            acc = self.latest_accel[0]
            self.get_logger().info(
                f'📈 Latest: Accel({acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}) m/s²'
            )
        
        if self.latest_gyro:
            gyr = self.latest_gyro[0]
            self.get_logger().info(
                f'🔄 Gyro({gyr[0]:.2f}, {gyr[1]:.2f}, {gyr[2]:.2f}) rad/s'
            )
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()
        self.get_logger().info('🛑 HYPERIMU Bridge shutdown complete')

def main(args=None):
    rclpy.init(args=args)
    node = HyperIMUBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
