#!/usr/bin/env python3
import rospy
import numpy as np
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ekf import EKF
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D

def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class FusionNode:
    def __init__(self):
        rospy.init_node("fusion_node")
        q0 = np.array([0.0, 0.0, 0.0])
        P0 = np.eye(3) * 0.1
        Q = np.diag([0.01, 0.01])
        R = np.diag([0.5, 0.5])
        self.ekf = EKF(q0, P0, Q, R)
        self.last_odom = None
        self.virtual_landmark = np.array([0.0, 0.0])
        rospy.Subscriber("/odometry", Odometry, self.odom_cb)
        rospy.Subscriber("/visual_pose", Pose2D, self.vision_cb)
        self.pub = rospy.Publisher("/fused_pose", Pose2D, queue_size=10)
        rospy.loginfo("[fusion_node] Gestart")

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z+q.x*q.y), 1.0-2.0*(q.y*q.y+q.z*q.z))
        if self.last_odom is None:
            self.last_odom = (x, y, yaw)
            return
        dx = x - self.last_odom[0]
        dy = y - self.last_odom[1]
        dtheta = wrap_angle(yaw - self.last_odom[2])
        dX = math.sqrt(dx**2 + dy**2)
        self.last_odom = (x, y, yaw)
        self.ekf.predict(dX, dtheta)
        self._publish()

    def vision_cb(self, msg):
        dist = math.sqrt(msg.x**2 + msg.y**2)
        bearing = math.atan2(msg.y, msg.x)
        if dist < 0.01:
            return
        z = np.array([dist, bearing])
        self.ekf.update(z, self.virtual_landmark)
        self._publish()

    def _publish(self):
        q = self.ekf.q
        msg = Pose2D()
        msg.x = q[0]
        msg.y = q[1]
        msg.theta = q[2]
        self.pub.publish(msg)

if __name__ == "__main__":
    node = FusionNode()
    rospy.spin()
