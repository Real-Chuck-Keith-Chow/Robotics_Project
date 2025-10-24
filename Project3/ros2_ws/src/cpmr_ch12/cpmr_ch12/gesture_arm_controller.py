#!/usr/bin/env python3
import os
import sys
import cv2
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from kinova_gen3_interfaces.srv import Status, SetTool

kinova_interface_path = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', '..',
    'kinova', 'kinova_ws', 'install', 'kinova_gen3_interfaces',
    'local', 'lib', 'python3.10', 'site-packages'
)
if os.path.exists(kinova_interface_path):
    sys.path.insert(0, kinova_interface_path)

class GestureArmControl(Node):
    BODY_PART_NAMES = [
        "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ]

    INDEX_NOSE = 0
    INDEX_LEFT_EYE = 1
    INDEX_RIGHT_EYE = 2
    INDEX_LEFT_SHOULDER = 5
    INDEX_RIGHT_SHOULDER = 6
    INDEX_LEFT_WRIST = 9
    INDEX_RIGHT_WRIST = 10

    def __init__(self):
        super().__init__('gesture_arm_control')

        model_path = os.path.join(get_package_share_directory('cpmr_ch12'), 'yolov8n-pose.pt')
        self.declare_parameter("model", model_path)
        model_file = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("device", "cpu")
        self.processing_device = self.get_parameter("device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.5)
        self.confidence_threshold = self.get_parameter("threshold").get_parameter_value().double_value

        self.declare_parameter("camera_topic", "/mycamera/image_raw")
        self.camera_topic_name = self.get_parameter("camera_topic").get_parameter_value().string_value

        self.arm_busy = False
        self.arm_position_current = [0.0, 0.20, 0.10]
        self.arm_position_target = [0.0, 0.20, 0.10]
        self.system_ready = False

        self.bridge = CvBridge()
        self.pose_model = YOLO(model_file)
        self.pose_model.fuse()

        self.home_service_client = self.create_client(Status, '/home')
        self.tool_service_client = self.create_client(SetTool, '/set_tool')

        self.get_logger().info('Waiting for Kinova services...')
        while not self.home_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /home service...')
        while not self.tool_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /set_tool service...')

        self.camera_subscription = self.create_subscription(
            Image,
            self.camera_topic_name,
            self.camera_callback,
            1
        )

        self.get_logger().info('Gesture Arm Control Node started')
        self.get_logger().info('Moving to home position...')
        self.move_to_home()

    def extract_keypoints(self, pose_results):
        keypoint_data = {}
        if not pose_results.keypoints:
            return keypoint_data
        for person_points in pose_results.keypoints:
            if person_points.conf is None:
                continue
            for part_id, (coord, conf) in enumerate(zip(person_points.xy[0], person_points.conf[0])):
                if conf >= self.confidence_threshold:
                    keypoint_data[part_id] = {
                        'x': float(coord[0]),
                        'y': float(coord[1]),
                        'confidence': float(conf),
                        'name': self.BODY_PART_NAMES[part_id]
                    }
        return keypoint_data

    def compute_reference_distance(self, keypoints):
        avg_eye_y = None
        if self.INDEX_LEFT_EYE in keypoints and self.INDEX_RIGHT_EYE in keypoints:
            avg_eye_y = (keypoints[self.INDEX_LEFT_EYE]['y'] + keypoints[self.INDEX_RIGHT_EYE]['y']) / 2.0
        elif self.INDEX_LEFT_EYE in keypoints:
            avg_eye_y = keypoints[self.INDEX_LEFT_EYE]['y']
        elif self.INDEX_RIGHT_EYE in keypoints:
            avg_eye_y = keypoints[self.INDEX_RIGHT_EYE]['y']
        elif self.INDEX_NOSE in keypoints:
            avg_eye_y = keypoints[self.INDEX_NOSE]['y']
        else:
            return None

        avg_shoulder_y = None
        if self.INDEX_LEFT_SHOULDER in keypoints and self.INDEX_RIGHT_SHOULDER in keypoints:
            avg_shoulder_y = (keypoints[self.INDEX_LEFT_SHOULDER]['y'] + keypoints[self.INDEX_RIGHT_SHOULDER]['y']) / 2.0
        elif self.INDEX_LEFT_SHOULDER in keypoints:
            avg_shoulder_y = keypoints[self.INDEX_LEFT_SHOULDER]['y']
        elif self.INDEX_RIGHT_SHOULDER in keypoints:
            avg_shoulder_y = keypoints[self.INDEX_RIGHT_SHOULDER]['y']
        else:
            return None

        return abs(avg_shoulder_y - avg_eye_y)

    def detect_hand_position(self, keypoints, wrist_index, shoulder_index, distance_threshold):
        if wrist_index not in keypoints or shoulder_index not in keypoints:
            return None
        wrist_y = keypoints[wrist_index]['y']
        shoulder_y = keypoints[shoulder_index]['y']
        vertical_offset = wrist_y - shoulder_y
        if vertical_offset < -distance_threshold:
            return "above"
        elif vertical_offset > distance_threshold:
            return "below"
        else:
            return "neutral"

    def interpret_gesture(self, keypoints):
        if not self.system_ready or self.arm_busy:
            return
        distance_threshold = self.compute_reference_distance(keypoints)
        if distance_threshold is None:
            self.get_logger().debug('Unable to compute distance threshold')
            return
        left_hand_state = self.detect_hand_position(keypoints, self.INDEX_LEFT_WRIST, self.INDEX_LEFT_SHOULDER, distance_threshold)
        right_hand_state = self.detect_hand_position(keypoints, self.INDEX_RIGHT_WRIST, self.INDEX_RIGHT_SHOULDER, distance_threshold)
        self.get_logger().info(f'Left hand: {left_hand_state}, Right hand: {right_hand_state} (threshold: {distance_threshold:.1f}px)')
        target_coords = None
        if left_hand_state == "above" and right_hand_state == "above":
            target_coords = (0.0, 0.20, 0.10)
            self.get_logger().info('Gesture: Both hands up')
        elif left_hand_state == "above" and right_hand_state != "above":
            target_coords = (0.10, 0.20, 0.10)
            self.get_logger().info('Gesture: Left hand up')
        elif left_hand_state == "below" and right_hand_state != "below":
            target_coords = (-0.10, 0.20, 0.10)
            self.get_logger().info('Gesture: Left hand down')
        elif right_hand_state == "above" and left_hand_state != "above":
            target_coords = (0.0, 0.10, 0.10)
            self.get_logger().info('Gesture: Right hand up')
        elif right_hand_state == "below" and left_hand_state != "below":
            target_coords = (0.0, 0.30, 0.10)
            self.get_logger().info('Gesture: Right hand down')
        if target_coords:
            self.move_to_coordinates(*target_coords)

    def move_to_home(self):
        self.arm_busy = True
        home_request = Status.Request()
        home_future = self.home_service_client.call_async(home_request)
        home_future.add_done_callback(self.on_home_response)

    def on_home_response(self, home_future):
        try:
            response = home_future.result()
            if response.status:
                self.get_logger().info('Home position reached')
                self.arm_busy = False
                self.get_logger().info('Moving to ready position...')
                self.move_to_coordinates(0.0, 0.20, 0.10)
                self.system_ready = True
            else:
                self.get_logger().error('Home movement failed')
                self.arm_busy = False
        except Exception as error:
            self.get_logger().error(f'Home call error: {error}')
            self.arm_busy = False

    def move_to_coordinates(self, pos_x, pos_y, pos_z):
        if self.arm_busy:
            return
        self.arm_busy = True
        tool_request = SetTool.Request()
        tool_request.x = float(pos_x)
        tool_request.y = float(pos_y)
        tool_request.z = float(pos_z)
        tool_request.theta_x = 180.0
        tool_request.theta_y = 0.0
        tool_request.theta_z = 0.0
        self.get_logger().info(f'Moving arm to ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})')
        self.arm_position_target = [pos_x, pos_y, pos_z]
        move_future = self.tool_service_client.call_async(tool_request)
        move_future.add_done_callback(self.on_move_response)

    def on_move_response(self, move_future):
        try:
            result = move_future.result()
            if result and result.status:
                self.get_logger().info('Move complete')
                self.arm_position_current = self.arm_position_target
            else:
                self.get_logger().error('Move command failed')
        except Exception as error:
            self.get_logger().error(f'SetTool service error: {error}')
        finally:
            self.arm_busy = False

    def camera_callback(self, image_msg):
        frame = self.bridge.imgmsg_to_cv2(image_msg)
        inference_results = self.pose_model.predict(
            source=frame,
            verbose=False,
            stream=False,
            conf=self.confidence_threshold,
            device=self.processing_device
        )
        if len(inference_results) != 1:
            return
        inference_results = inference_results[0].cpu()
        if len(inference_results.boxes.data) == 0:
            return
        keypoints = self.extract_keypoints(inference_results)
        if len(keypoints) > 0:
            self.interpret_gesture(keypoints)
            annotated_image = inference_results.plot()
            if not self.system_ready:
                status_label = "INITIALIZING"
                status_color = (255, 255, 0)
            elif self.arm_busy:
                status_label = "MOVING"
                status_color = (0, 0, 255)
            else:
                status_label = "READY"
                status_color = (0, 255, 0)
            cv2.putText(annotated_image, status_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            position_text = f"Arm: ({self.arm_position_current[0]:.2f}, {self.arm_position_current[1]:.2f}, {self.arm_position_current[2]:.2f})"
            cv2.putText(annotated_image, position_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ref_distance = self.compute_reference_distance(keypoints) or 0
            left_state = self.detect_hand_position(keypoints, self.INDEX_LEFT_WRIST, self.INDEX_LEFT_SHOULDER, ref_distance)
            right_state = self.detect_hand_position(keypoints, self.INDEX_RIGHT_WRIST, self.INDEX_RIGHT_SHOULDER, ref_distance)
            hand_status_text = f"L: {left_state or 'N/A'}  R: {right_state or 'N/A'}"
            cv2.putText(annotated_image, hand_status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            threshold_text = f"Threshold: {ref_distance:.1f}px"
            cv2.putText(annotated_image, threshold_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            note_text = "Note: Anatomical left/right (not mirrored)"
            cv2.putText(annotated_image, note_text, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            instruction_text = "Raise/lower hands beyond threshold to control"
            cv2.putText(annotated_image, instruction_text, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.imshow('Gesture Arm Control', annotated_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    gesture_node = GestureArmControl()
    try:
        rclpy.spin(gesture_node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

