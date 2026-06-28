#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
from typing import Optional
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge
from std_msgs.msg import Bool
# -- Rijparameters --
VELOCITY      = 0.25   # voorwaartse snelheid (m/s)
KP            = 0.035  # proportionele gain
KI            = 0.0005 # integrale gain (corrigeert structurele afwijking)
KD            = 0.010  # afgeleide gain (dempt oscillaties)
# -- ROI: onderste deel van het frame --
ROI_FRACTION  = 0.5
# -- HSV: geel (middenlijn) --
YELLOW_LOW  = np.array([ 20,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([ 35, 255, 255], dtype=np.uint8)
MIN_CONTOUR_AREA = 300
class LaneFollowingNode(DTROS):
    """
    Lane-following: volgt de gele MIDDENLIJN. De bot stuurt zo dat zijn
    midden op de gele lijn blijft, via een PID-regelaar.
    Pauzeert bij /duckie_stop of /intersection_active.
    """
    def __init__(self, node_name: str):
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        vehicle_name = os.environ["VEHICLE_NAME"]
        camera_topic = f"/{vehicle_name}/camera_node/image/compressed"
        twist_topic  = f"/{vehicle_name}/car_cmd_switch_node/cmd"
        self._bridge = CvBridge()
        self._duckie_stop = False
        self._intersection_active = False
        self._prev_error = 0.0
        self._integral = 0.0
        self._prev_time  = rospy.Time.now()
        self._pub = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self._sub = rospy.Subscriber(camera_topic, CompressedImage, self._cb_image, queue_size=1)
        stop_topic = f"/{vehicle_name}/duckie_stop"
        self._stop_sub = rospy.Subscriber(stop_topic, Bool, self._cb_stop, queue_size=1)
        self._inter_sub = rospy.Subscriber("/intersection_active", Bool, self._cb_intersection, queue_size=1)
        rospy.loginfo(f"[{node_name}] Gestart - volgt de gele middenlijn op {camera_topic}")
    def _preprocess(self, bgr_image):
        small = cv2.resize(bgr_image, (320, 240))
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return small, hsv
    def _get_roi(self, image):
        h = image.shape[0]
        start_row = int(h * (1.0 - ROI_FRACTION))
        return image[start_row:, :]
    def _largest_centroid_x(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        return M["m10"] / M["m00"]
    def _cb_stop(self, msg):
        self._duckie_stop = msg.data
    def _cb_intersection(self, msg):
        self._intersection_active = msg.data
    def _cb_image(self, msg):
        if self._duckie_stop:
            self._publish(v=0.0, omega=0.0)
            rospy.logwarn_throttle(1.0, "DUCKIE GEDETECTEERD - robot gestopt!")
            return
        if self._intersection_active:
            rospy.loginfo_throttle(1.0, "Kruispunt actief - navigatie-node bestuurt.")
            return
        bgr  = self._bridge.compressed_imgmsg_to_cv2(msg)
        _, hsv = self._preprocess(bgr)
        hsv_roi = self._get_roi(hsv)
        width   = hsv_roi.shape[1]
        cx      = width / 2.0
        mask_yellow = cv2.inRange(hsv_roi, YELLOW_LOW, YELLOW_HIGH)
        x_yellow = self._largest_centroid_x(mask_yellow)
        if x_yellow is not None:
            lane_center = x_yellow
        else:
            self._publish(v=0.0, omega=0.0)
            rospy.logwarn_throttle(2.0, "Geen gele lijn gevonden - robot gestopt.")
            return
        error = lane_center - cx
        now  = rospy.Time.now()
        dt   = (now - self._prev_time).to_sec()
        dt   = max(dt, 1e-3)
        self._integral += error * dt
        self._integral = max(min(self._integral, 500.0), -500.0)  # tegen integral windup
        derivative   = (error - self._prev_error) / dt
        omega        = -(KP * error + KI * self._integral + KD * derivative)
        self._prev_error = error
        self._prev_time  = now
        self._publish(v=VELOCITY, omega=omega)
        y_str = f"{x_yellow:.0f}" if x_yellow is not None else "N/A"
        stuur = "LINKS" if omega > 0.05 else ("RECHTS" if omega < -0.05 else "RECHTUIT")
        rospy.loginfo_throttle(1.0,
            f"\n--- Lane Following (gele middenlijn) ---\n"
            f"  Gele lijn (x)    : {y_str} px\n"
            f"  Beeldmidden (cx) : {cx:.0f} px\n"
            f"  Laterale fout    : {error:+.1f} px\n"
            f"  Stuurhoek (omega): {omega:+.3f} rad/s\n"
            f"  Richting         : {stuur}\n"
            f"----------------------")
    def _publish(self, v, omega):
        msg = Twist2DStamped()
        msg.v = v
        msg.omega = omega
        self._pub.publish(msg)
    def on_shutdown(self):
        self._publish(v=0.0, omega=0.0)
        rospy.loginfo("Lane following gestopt - robot tot stilstand gebracht.")
if __name__ == "__main__":
    node = LaneFollowingNode(node_name="lane_following_node")
    rospy.spin()
