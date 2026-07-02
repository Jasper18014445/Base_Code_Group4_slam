#!/usr/bin/env python3

import rospy
import math
from duckietown_msgs.msg import WheelEncoderStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import tf.transformations as tft
import os

class SimpleOdometryNode:

    def __init__(self):
        rospy.init_node("simple_odometry_node")

        self.vehicle = os.environ.get("VEHICLE_NAME")
        if self.vehicle is None:
            rospy.logerr("VEHICLE_NAME not set")
            raise RuntimeError()

        left_topic = f"/{self.vehicle}/left_wheel_encoder_driver_node/tick"
        right_topic = f"/{self.vehicle}/right_wheel_encoder_driver_node/tick"

        rospy.Subscriber(left_topic, WheelEncoderStamped, self.left_cb)
        rospy.Subscriber(right_topic, WheelEncoderStamped, self.right_cb)

        self.pub = rospy.Publisher("odometry", Odometry, queue_size=10)

        # state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_left = None
        self.last_right = None

        # robot parameters (adjust if needed)
        self.wheel_radius = 0.0318
        self.baseline = 0.1

    def left_cb(self, msg):
        self.last_left = msg.data
        self.compute()

    def right_cb(self, msg):
        self.last_right = msg.data
        self.compute()

    def compute(self):
        if self.last_left is None or self.last_right is None:
            return

        # very simplified differential drive model
        dl = self.last_left * self.wheel_radius
        dr = self.last_right * self.wheel_radius

        v = (dr + dl) / 2.0
        w = (dr - dl) / self.baseline

        dt = 0.1  # assume fixed timestep (simple version)

        self.theta += w * dt
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        q = tft.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation = Quaternion(*q)

        self.pub.publish(odom)


if __name__ == "__main__":
    node = SimpleOdometryNode()
    rospy.spin()