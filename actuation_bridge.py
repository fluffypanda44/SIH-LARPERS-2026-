"""
Urban Search & Rescue (USAR) Actuation Dispatcher & Robotics Bridge.
Converts autonomous navigation directives into standard ROS2 geometry_msgs/Twist
telemetry packets and broadcasts them over UDP/Serial to robot motor controllers.
"""

import json
import math
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Twist:
    linear: Vector3
    angular: Vector3


@dataclass
class RobotTelemetryPacket:
    timestamp: float
    mode: str
    twist: Dict[str, Dict[str, float]]
    emergency_brake: bool
    action_text: str
    target_heading_deg: float
    target_distance_m: float


class ActuationBridge:
    """
    Bridges AI perception decisions to physical or simulated robot motor drivers.
    Broadcasts standard ROS2-compliant velocity packets via UDP datagrams.
    """

    def __init__(self, udp_ip: str = "127.0.0.1", udp_port: int = 9090, max_angular_rad_s: float = 1.2):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.max_angular = max_angular_rad_s

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception:
            pass

        self.last_packet: Optional[RobotTelemetryPacket] = None
        self.packets_sent = 0
        print(f"[ACTUATION] Bridge initialized. Broadcasting ROS2 Twist to UDP {self.udp_ip}:{self.udp_port}")

    def dispatch(self, nav_command: Any) -> RobotTelemetryPacket:
        """
        Converts AutonomousNavCommand to standard Twist command.
        
        Kinematics:
        - linear.x: Forward velocity (m/s)
        - angular.z: Yaw rate (rad/s). Negative = Turn Right, Positive = Turn Left
        """
        heading_deg = getattr(nav_command, "target_heading_deg", 0.0)
        recommended_speed = getattr(nav_command, "recommended_speed", 0.0)
        mode = getattr(nav_command, "mode", "UNKNOWN")
        action = getattr(nav_command, "action_text", "")
        dist = getattr(nav_command, "target_distance_m", 0.0)

        emergency_brake = (mode == "RESCUE_STATIONARY" or recommended_speed <= 0.0)

        if emergency_brake:
            linear_x = 0.0
            angular_z = 0.0
        else:
            linear_x = float(recommended_speed)
            # Proportional steering controller: heading_deg to angular velocity in rad/s
            heading_rad = math.radians(heading_deg)
            # Standard ROS REP-103: positive angular.z = counter-clockwise (Left)
            angular_z = float(-heading_rad * 1.5)
            angular_z = max(-self.max_angular, min(self.max_angular, angular_z))

        twist_dict = {
            "linear": {"x": round(linear_x, 3), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": round(angular_z, 3)},
        }

        packet = RobotTelemetryPacket(
            timestamp=time.time(),
            mode=mode,
            twist=twist_dict,
            emergency_brake=emergency_brake,
            action_text=action,
            target_heading_deg=heading_deg,
            target_distance_m=dist,
        )

        # Broadcast packet over UDP
        try:
            payload = json.dumps(asdict(packet)).encode("utf-8")
            self.sock.sendto(payload, (self.udp_ip, self.udp_port))
            self.packets_sent += 1
        except Exception:
            pass

        self.last_packet = packet
        return packet

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
