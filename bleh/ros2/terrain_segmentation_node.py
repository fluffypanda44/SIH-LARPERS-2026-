"""
ROS 2 Node for Real-Time Off-Road Terrain Traversability Segmentation.

Subscribes:
  - /camera/left/image_raw (sensor_msgs/msg/Image)

Publishes:
  - /terrain/segmentation_mask (sensor_msgs/msg/Image, mono8: 0..3)
  - /terrain/traversability_costmap (sensor_msgs/msg/Image, mono8: 0..254)
  - /terrain/colored_overlay (sensor_msgs/msg/Image, rgb8 for RViz2 display)
"""

import sys
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image as RosImage
    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False

import torch
from PIL import Image

# Import local modules
try:
    from models import build_model
    from data import mask_to_color, mask_to_nav2_costmap
except ImportError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models import build_model
    from data import mask_to_color, mask_to_nav2_costmap


class TerrainSegmentationNode:
    """
    Terrain Traversability Segmentation Node.
    Defined conditionally so it can be imported or executed in both ROS 2
    and simulated standalone environments.
    """

    def __init__(self, node_instance=None):
        self.node = node_instance
        self.target_height = 512
        self.target_width = 1024
        self.num_classes = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        # Initialize Model
        self.model = build_model("fast_scnn", num_classes=self.num_classes).to(self.device)
        self.model.eval()

        if self.node is not None:
            self._init_ros()

    def _init_ros(self):
        self.node.declare_parameter("weights_path", "")
        self.node.declare_parameter("input_height", 512)
        self.node.declare_parameter("input_width", 1024)
        self.node.declare_parameter("publish_overlay", True)

        weights_path = self.node.get_parameter("weights_path").get_parameter_value().string_value
        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            state_dict = state.get("model_state_dict", state)
            self.model.load_state_dict(state_dict)
            self.node.get_logger().info(f"Loaded Fast-SCNN weights from {weights_path}")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers & Publishers
        self.sub_image = self.node.create_subscription(
            RosImage,
            "/camera/left/image_raw",
            self.image_callback,
            qos_profile,
        )
        self.pub_mask = self.node.create_publisher(RosImage, "/terrain/segmentation_mask", 10)
        self.pub_costmap = self.node.create_publisher(RosImage, "/terrain/traversability_costmap", 10)
        self.pub_overlay = self.node.create_publisher(RosImage, "/terrain/colored_overlay", 10)

        self.node.get_logger().info("TerrainSegmentationNode initialized and listening to /camera/left/image_raw")

    def process_frame(self, rgb_image: np.ndarray):
        """
        Runs neural inference on RGB NumPy frame.
        Input: (H, W, 3) uint8 RGB
        Returns:
            pred_mask: (H, W) uint8 class mask (0..3)
            costmap: (H, W) uint8 Nav2 costmap (0..254)
            overlay: (H, W, 3) uint8 blended RGB image
        """
        h_orig, w_orig = rgb_image.shape[:2]
        pil_img = Image.fromarray(rgb_image).resize((self.target_width, self.target_height), Image.BILINEAR)

        # Preprocessing
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))
        img_tensor = np.expand_dims(img_arr, axis=0)
        normalized = (img_tensor - self.mean) / self.std

        # Inference
        with torch.no_grad():
            tensor_in = torch.from_numpy(normalized).float().to(self.device)
            logits = self.model(tensor_in)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        # Costmap & Overlay
        costmap = mask_to_nav2_costmap(pred_mask)
        color_mask = mask_to_color(pred_mask)
        overlay = (0.5 * np.array(pil_img, dtype=np.float32) + 0.5 * color_mask.astype(np.float32)).clip(0, 255).astype(np.uint8)

        return pred_mask, costmap, overlay

    def image_callback(self, msg: RosImage):
        """ROS 2 message callback."""
        # Unpack raw image data
        if msg.encoding in ("rgb8", "bgr8"):
            channels = 3
        elif msg.encoding == "mono8":
            channels = 1
        else:
            channels = 3

        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
        if msg.encoding == "bgr8":
            img_np = img_np[:, :, ::-1]  # Convert BGR to RGB

        pred_mask, costmap, overlay = self.process_frame(img_np)

        # Publish results
        mask_msg = RosImage()
        mask_msg.header = msg.header
        mask_msg.height, mask_msg.width = pred_mask.shape
        mask_msg.encoding = "mono8"
        mask_msg.step = pred_mask.shape[1]
        mask_msg.data = pred_mask.tobytes()
        self.pub_mask.publish(mask_msg)

        cost_msg = RosImage()
        cost_msg.header = msg.header
        cost_msg.height, cost_msg.width = costmap.shape
        cost_msg.encoding = "mono8"
        cost_msg.step = costmap.shape[1]
        cost_msg.data = costmap.tobytes()
        self.pub_costmap.publish(cost_msg)

        overlay_msg = RosImage()
        overlay_msg.header = msg.header
        overlay_msg.height, overlay_msg.width = overlay.shape[:2]
        overlay_msg.encoding = "rgb8"
        overlay_msg.step = overlay.shape[1] * 3
        overlay_msg.data = overlay.tobytes()
        self.pub_overlay.publish(overlay_msg)


def main():
    if not HAVE_ROS2:
        print("ROS 2 (rclpy) is not installed in this environment.")
        print("This node is ready to be launched inside your ROS 2 workspace on the robot or Jetson Orin.")
        sys.exit(0)

    rclpy.init()
    node = Node("terrain_segmentation_node")
    _ = TerrainSegmentationNode(node_instance=node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
