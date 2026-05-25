# Xihe,
# First submit on May, 11, 2026

import time
from collections import deque
from threading import Lock

import numpy as np

from fluxvla.engines.utils.root import OPERATORS


@OPERATORS.register_module()
class ARXOperator:
    """ARX operator for ROS-based robotic arm control.

    This class handles robot arm control, sensor data collection, and
    synchronization for ARX arms in a ROS environment.
    Supports RGB and depth image streams, joint states, and gripper control.
    """

    def __init__(self,
                 img_top_topic,
                 img_front_topic,
                 arm_status_topic,
                 arm_cmd_topic,
                 use_depth_image=False,
                 img_top_depth_topic=None,
                 img_front_depth_topic=None):
        """Initialize ARXOperator with ROS topics configuration.

        Args:
            img_top_topic (str): ROS topic for top camera RGB image
            img_front_topic (str): ROS topic for front camera RGB image
            arm_status_topic (str): ROS topic for arm status (joint states)
            arm_cmd_topic (str): ROS topic for arm command
            use_depth_image (bool, optional): Whether to use depth images.
                Defaults to False.
            img_top_depth_topic (str, optional): ROS topic for top depth
                image. Required when use_depth_image=True.
            img_front_depth_topic (str, optional): ROS topic for front depth
                image. Required when use_depth_image=True.

        Raises:
            ValueError: When use_depth_image=True but depth topics not provided
        """
        self.img_top_topic = img_top_topic
        self.img_front_topic = img_front_topic
        self.arm_status_topic = arm_status_topic
        self.arm_cmd_topic = arm_cmd_topic
        self.use_depth_image = use_depth_image
        self.img_top_depth_topic = img_top_depth_topic
        self.img_front_depth_topic = img_front_depth_topic

        if self.use_depth_image:
            if not img_top_depth_topic or not img_front_depth_topic:
                raise ValueError(
                    'When use_depth_image=True, both img_top_depth_topic '
                    'and img_front_depth_topic must be provided')

        self._init_count()
        self._init()
        self._init_ros()

    def _init_count(self):
        """Initialize error counters for different data streams."""
        self.rgb_top_count = 0
        self.rgb_front_count = 0
        self.depth_top_count = 0
        self.depth_front_count = 0
        self.arm_status_count = 0

    def _init(self):
        """Initialize internal data structures and OpenCV bridge."""
        from cv_bridge import CvBridge

        self.rgb_t = 0
        self.rgb_f = 0
        self.depth_t = 0
        self.depth_f = 0
        self.arm_status_err = 0

        self.last_time_step = 0
        self.bridge = CvBridge()

        self.img_top_deque = deque()
        self.img_front_deque = deque()
        self.img_top_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.arm_status_deque = deque()

        self._arm_cmd_lock = Lock()
        self._latest_joint_pos = None
        self._latest_gripper_pos = None

    def _init_ros(self):
        """Initialize ROS subscribers and publishers."""
        import rospy
        from arx5_arm_msg.msg import RobotCmd, RobotStatus
        from sensor_msgs.msg import Image

        self._img_top_sub = rospy.Subscriber(
            self.img_top_topic, Image, self._img_top_callback, queue_size=1)
        self._img_front_sub = rospy.Subscriber(
            self.img_front_topic,
            Image,
            self._img_front_callback,
            queue_size=1)

        if self.use_depth_image:
            self._img_top_depth_sub = rospy.Subscriber(
                self.img_top_depth_topic,
                Image,
                self._img_top_depth_callback,
                queue_size=1)
            self._img_front_depth_sub = rospy.Subscriber(
                self.img_front_depth_topic,
                Image,
                self._img_front_depth_callback,
                queue_size=1)

        self._arm_status_sub = rospy.Subscriber(
            self.arm_status_topic,
            RobotStatus,
            self._arm_status_callback,
            queue_size=1)

        self._arm_cmd_pub = rospy.Publisher(
            self.arm_cmd_topic, RobotCmd, queue_size=1)

    def _img_top_callback(self, msg):
        """Callback for top camera RGB image.

        Args:
            msg (sensor_msgs.msg.Image): RGB image message
        """
        self.img_top_deque.append(msg)

    def _img_front_callback(self, msg):
        """Callback for front camera RGB image.

        Args:
            msg (sensor_msgs.msg.Image): RGB image message
        """
        self.img_front_deque.append(msg)

    def _img_top_depth_callback(self, msg):
        """Callback for top camera depth image.

        Args:
            msg (sensor_msgs.msg.Image): Depth image message
        """
        self.img_top_depth_deque.append(msg)

    def _img_front_depth_callback(self, msg):
        """Callback for front camera depth image.

        Args:
            msg (sensor_msgs.msg.Image): Depth image message
        """
        self.img_front_depth_deque.append(msg)

    def _arm_status_callback(self, msg):
        """Callback for arm status message.

        Args:
            msg (arx5_arm_msg.msg.RobotStatus): Arm status message
        """
        self.arm_status_deque.append(msg)

    def get_frame(self, slop=0.7):
        """Get synchronized frame data from all sensors.

        Synchronizes RGB images, depth images (if enabled), joint states,
        and gripper states based on timestamps.

        Args:
            slop (float, optional): Maximum allowed time difference between
                sensors in seconds. Defaults to 0.7.

        Returns:
            tuple or False: If successful, returns tuple containing:
                (img_top, img_front, img_top_depth, img_front_depth,
                 arm_joint_pos, arm_gripper, frame_time, frame_time_max)
                If failed, returns False.
        """
        required_queues_empty = (
            len(self.img_top_deque) == 0 or len(self.img_front_deque) == 0)

        depth_queues_empty = (
            self.use_depth_image and (len(self.img_top_depth_deque) == 0
                                      or len(self.img_front_depth_deque) == 0))

        if required_queues_empty or depth_queues_empty:
            self._handle_empty_queues()
            return False

        if len(self.arm_status_deque) == 0:
            self._handle_arm_status_empty()
            return False

        frame_time = self._calculate_frame_time()

        if not self._check_sensor_data_availability(frame_time):
            return False

        self.last_time_step = frame_time

        self.rgb_t = 0
        self.rgb_f = 0
        self.depth_t = 0
        self.depth_f = 0
        self.arm_status_err = 0

        frame_time_max = self._synchronize_queues(frame_time)

        if abs(frame_time_max - frame_time) > slop:
            self._flush_outdated_data(frame_time)
            return False

        return self._extract_synchronized_data()

    def _handle_empty_queues(self):
        """Handle empty data queues by incrementing error counters."""
        if len(self.img_top_deque) == 0:
            self.rgb_t += 1
            if self.rgb_t > 3:
                print('Error top RGB', str(time.time()))

        if len(self.img_front_deque) == 0:
            self.rgb_f += 1
            if self.rgb_f > 3:
                print('Error front RGB', str(time.time()))

        if self.use_depth_image:
            if len(self.img_top_depth_deque) == 0:
                self.depth_t += 1
                if self.depth_t > 3:
                    print('Error top Depth')

            if len(self.img_front_depth_deque) == 0:
                self.depth_f += 1
                if self.depth_f > 3:
                    print('Error front Depth')

    def _handle_arm_status_empty(self):
        """Handle empty arm status queue."""
        self.arm_status_err += 1
        if self.arm_status_err > 3:
            print('Error arm status', str(time.time()))

    def _calculate_frame_time(self):
        """Calculate the minimum frame time across all sensors.

        Returns:
            float: Minimum timestamp across all available sensors
        """
        timestamps = [
            self.img_top_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec()
        ]

        if self.use_depth_image:
            timestamps.extend([
                self.img_top_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec()
            ])

        timestamps.append(self.arm_status_deque[-1].header.stamp.to_sec())

        return min(timestamps)

    def _check_sensor_data_availability(self, frame_time):
        """Check if all sensors have data at the given frame time.

        Args:
            frame_time (float): Target frame timestamp

        Returns:
            bool: True if all sensors have data, False otherwise
        """
        top_img_time = self.img_top_deque[-1].header.stamp.to_sec()
        front_img_time = self.img_front_deque[-1].header.stamp.to_sec()
        arm_time = self.arm_status_deque[-1].header.stamp.to_sec()

        if abs(top_img_time - frame_time) > 1.0 or \
           abs(front_img_time - frame_time) > 1.0 or \
           abs(arm_time - frame_time) > 1.0:
            return False

        if self.use_depth_image:
            top_depth_time = self.img_top_depth_deque[-1].header.stamp.to_sec()
            front_depth_time = self.img_front_depth_deque[
                -1].header.stamp.to_sec()
            if abs(top_depth_time - frame_time) > 1.0 or \
               abs(front_depth_time - frame_time) > 1.0:
                return False

        return True

    def _synchronize_queues(self, frame_time):
        """Synchronize all data queues to the given frame time.

        Args:
            frame_time (float): Target frame timestamp

        Returns:
            float: Maximum timestamp after synchronization
        """
        while self.img_top_deque and \
                self.img_top_deque[0].header.stamp.to_sec() \
                < frame_time - 0.5:
            self.img_top_deque.popleft()

        while self.img_front_deque and \
                self.img_front_deque[0].header.stamp.to_sec() \
                < frame_time - 0.5:
            self.img_front_deque.popleft()

        while self.arm_status_deque and \
                self.arm_status_deque[0].header.stamp.to_sec() \
                < frame_time - 0.5:
            self.arm_status_deque.popleft()

        if self.use_depth_image:
            while self.img_top_depth_deque and \
                    self.img_top_depth_deque[0].header.stamp.to_sec() \
                    < frame_time - 0.5:
                self.img_top_depth_deque.popleft()

            while self.img_front_depth_deque and \
                    self.img_front_depth_deque[0].header.stamp.to_sec() \
                    < frame_time - 0.5:
                self.img_front_depth_deque.popleft()

        times = [frame_time]
        if self.img_top_deque:
            times.append(self.img_top_deque[-1].header.stamp.to_sec())
        if self.img_front_deque:
            times.append(self.img_front_deque[-1].header.stamp.to_sec())
        if self.arm_status_deque:
            times.append(self.arm_status_deque[-1].header.stamp.to_sec())

        return max(times)

    def _flush_outdated_data(self, frame_time):
        """Flush outdated data from all queues.

        Args:
            frame_time (float): Target frame timestamp
        """
        while self.img_top_deque and \
                self.img_top_deque[0].header.stamp.to_sec() \
                < frame_time - 1.0:
            self.img_top_deque.popleft()

        while self.img_front_deque and \
                self.img_front_deque[0].header.stamp.to_sec() \
                < frame_time - 1.0:
            self.img_front_deque.popleft()

        while self.arm_status_deque and \
                self.arm_status_deque[0].header.stamp.to_sec() \
                < frame_time - 1.0:
            self.arm_status_deque.popleft()

        if self.use_depth_image:
            while self.img_top_depth_deque and \
                    self.img_top_depth_deque[0].header.stamp.to_sec() \
                    < frame_time - 1.0:
                self.img_top_depth_deque.popleft()

            while self.img_front_depth_deque and \
                    self.img_front_depth_deque[0].header.stamp.to_sec() \
                    < frame_time - 1.0:
                self.img_front_depth_deque.popleft()

    def _extract_synchronized_data(self):
        """Extract synchronized data from all queues.

        Returns:
            tuple: Synchronized (img_top, img_front, img_top_depth,
                   img_front_depth, arm_joint_pos, arm_gripper,
                   frame_time, frame_time_max)
        """
        from std_msgs.msg import Header

        img_top = self.bridge.imgmsg_to_cv2(
            self.img_top_deque[-1], desired_encoding='bgr8')
        img_front = self.bridge.imgmsg_to_cv2(
            self.img_front_deque[-1], desired_encoding='bgr8')

        img_top_depth = None
        img_front_depth = None
        if self.use_depth_image:
            img_top_depth = self.bridge.imgmsg_to_cv2(
                self.img_top_depth_deque[-1], desired_encoding='passthrough')
            img_front_depth = self.bridge.imgmsg_to_cv2(
                self.img_front_depth_deque[-1], desired_encoding='passthrough')

        arm_status = self.arm_status_deque[-1]
        arm_joint_pos = np.array(arm_status.joint_pos)

        arm_gripper = 0.0
        if len(arm_status.joint_pos) > 6:
            arm_gripper = arm_status.joint_pos[6]

        class StampedFloat:

            def __init__(self, data, stamp):
                self.data = data
                self.header = Header(stamp=stamp)

        arm_gripper_stamped = StampedFloat(arm_gripper,
                                           arm_status.header.stamp)

        frame_time = arm_status.header.stamp.to_sec()
        frame_time_max = frame_time

        return (img_top, img_front, img_top_depth, img_front_depth,
                arm_joint_pos, arm_gripper_stamped, frame_time, frame_time_max)

    def send_joints(self, qpos):
        """Send joint positions to the robot.

        Args:
            qpos (np.ndarray): Joint positions (6 DOF)
        """
        import rospy
        from arx5_arm_msg.msg import RobotCmd

        with self._arm_cmd_lock:
            cmd = RobotCmd()
            cmd.header.stamp = rospy.Time.now()
            cmd.joint_pos = qpos.tolist()
            cmd.gripper = 0.0
            cmd.mode = 0
            self._arm_cmd_pub.publish(cmd)

    def send_gripper(self, gripper_position):
        """Send gripper position to the robot.

        Args:
            gripper_position (float): Gripper position
        """
        import rospy
        from arx5_arm_msg.msg import RobotCmd

        with self._arm_cmd_lock:
            if self._latest_joint_pos is not None:
                cmd = RobotCmd()
                cmd.header.stamp = rospy.Time.now()
                cmd.joint_pos = self._latest_joint_pos.tolist()
                cmd.gripper = gripper_position
                cmd.mode = 0
                self._arm_cmd_pub.publish(cmd)

    def movej(self, qpos):
        """Move robot to joint positions (blocking).

        Args:
            qpos (np.ndarray): Target joint positions
        """
        self.send_joints(qpos)

    def movel(self, eepose):
        """Move robot to end-effector pose (placeholder).

        Args:
            eepose (np.ndarray): Target end-effector pose
        """
        pass

    def movegrip(self, gripper_position):
        """Move gripper to target position.

        Args:
            gripper_position (float): Target gripper position
        """
        self.send_gripper(gripper_position)

    def servoj(self, qpos):
        """Send servo command to robot (same as send_joints).

        Args:
            qpos (np.ndarray): Joint positions
        """
        self.send_joints(qpos)
