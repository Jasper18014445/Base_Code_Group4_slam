#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D

from ekf import EKF  # jouw EKF bestand!


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


class FusionNode:
    def __init__(self):
        rospy.init_node("fusion_node")

        # EKF init
        q0 = np.array([0.0, 0.0, 0.0])
        P0 = np.eye(3) * 0.1
        Q = np.diag([0.01, 0.01])  # motion noise
        R = np.diag([0.5, 0.5])    # measurement noise

        self.ekf = EKF(q0, P0, Q, R)

        self.last_odom = None

        # Subscribers
        rospy.Subscriber("/odometry", Odometry, self.odom_cb)
        rospy.Subscriber("/visual_pose", Pose2D, self.vision_cb)

        # Publisher
        self.pub = rospy.Publisher("/fused_pose", Pose2D, queue_size=10)

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # yaw uit quaternion
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        if self.last_odom is None:
            self.last_odom = (x, y, yaw)
            return

        dx = x - self.last_odom[0]
        dy = y - self.last_odom[1]
        dtheta = wrap_angle(yaw - self.last_odom[2])

        dX = np.sqrt(dx**2 + dy**2)

        self.ekf.predict(dX, dtheta)

        self.last_odom = (x, y, yaw)

        self.publish()

    def vision_cb(self, msg):
        # Fake measurement: treat visual pose as landmark at (x,y)
        z = np.array([np.sqrt(msg.x**2 + msg.y**2), msg.theta])
        tag_xy = np.array([msg.x, msg.y])

        self.ekf.update(z, tag_xy)

        self.publish()

    def publish(self):
        q = self.ekf.q

        msg = Pose2D()
        msg.x = q[0]
        msg.y = q[1]
        msg.theta = q[2]

        self.pub.publish(msg)


if __name__ == "__main__":
    node = FusionNode()
    rospy.spin()