#!/usr/bin/env python3

import rospy
from duckietown_msgs.msg import WheelEncoderStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import tf
import math


class OdometryNode:
    def __init__(self):
        rospy.init_node('odometry_node')

        # Parameters (tune voor Duckiebot!)
        self.R = rospy.get_param("~wheel_radius", 0.0318)   # meters
        self.L = rospy.get_param("~wheel_baseline", 0.1)    # afstand tussen wielen

        self.ticks_per_rev = rospy.get_param("~ticks_per_rev", 135)

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.prev_left_ticks = None
        self.prev_right_ticks = None

        # Subscribers
        vehicle_name = rospy.get_param("~vehicle_name", "virtbot1")
        rospy.Subscriber(f"/{vehicle_name}/left_wheel_encoder_driver_node/tick", WheelEncoderStamped, self.left_cb)
        rospy.Subscriber(f"/{vehicle_name}/right_wheel_encoder_driver_node/tick", WheelEncoderStamped, self.right_cb)

        # Publisher
        self.odom_pub = rospy.Publisher("/odometry", Odometry, queue_size=10)

        self.left_ticks = 0
        self.right_ticks = 0

        self.last_time = rospy.Time.now()

    def left_cb(self, msg):
        self.left_ticks = msg.data
        self.update_odometry()

    def right_cb(self, msg):
        self.right_ticks = msg.data
        self.update_odometry()

    def update_odometry(self):
        if self.prev_left_ticks is None:
            self.prev_left_ticks = self.left_ticks
            self.prev_right_ticks = self.right_ticks
            return

        # Δ ticks
        d_left = self.left_ticks - self.prev_left_ticks
        d_right = self.right_ticks - self.prev_right_ticks

        self.prev_left_ticks = self.left_ticks
        self.prev_right_ticks = self.right_ticks

        # Omzetten naar afstand
        dist_per_tick = 2 * math.pi * self.R / self.ticks_per_rev

        d_left_m = d_left * dist_per_tick
        d_right_m = d_right * dist_per_tick

        # Differential drive
        d_s = (d_right_m + d_left_m) / 2.0
        d_theta = (d_right_m - d_left_m) / self.L

        # Tijd
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Update pose
        self.x += d_s * math.cos(self.theta)
        self.y += d_s * math.sin(self.theta)
        self.theta += d_theta

        # Quaternion
        quat = tf.transformations.quaternion_from_euler(0, 0, self.theta)

        # Odometry message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        odom.pose.pose.orientation = Quaternion(*quat)

        # Velocity (optioneel)
        if dt > 0:
            odom.twist.twist.linear.x = d_s / dt
            odom.twist.twist.angular.z = d_theta / dt

        self.odom_pub.publish(odom)


if __name__ == "__main__":
    node = OdometryNode()
    rospy.spin()
