#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge


class SLAMNode:
    def __init__(self):
        rospy.init_node('slam_node')

        self.bridge = CvBridge()

        # Subscribers
        rospy.Subscriber(
            "/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            queue_size=1
        )

        # Publisher
        self.pose_pub = rospy.Publisher("/visual_pose", Pose2D, queue_size=10)

        # ORB detector
        self.orb = cv2.ORB_create(1000)

        self.prev_gray = None
        self.prev_pts = None

        # Camera intrinsics (MOET je calibreren!)
        self.K = np.array([
            [300, 0, 160],
            [0, 300, 120],
            [0, 0, 1]
        ])

        # Pose (relatief)
        self.x = 0
        self.y = 0
        self.theta = 0

    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, 200, 0.01, 7)
            return

        # Track features
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None
        )

        good_prev = self.prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) < 8:
            # reset features
            self.prev_pts = cv2.goodFeaturesToTrack(gray, 200, 0.01, 7)
            self.prev_gray = gray
            return

        # Essential matrix
        E, _ = cv2.findEssentialMat(
            good_next, good_prev, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None:
            return

        _, R, t, _ = cv2.recoverPose(E, good_next, good_prev, self.K)

        # Extract yaw (2D benadering)
        d_theta = np.arctan2(R[1, 0], R[0, 0])

        # Scale probleem: t is onbekend geschaald!
        scale = 0.05  # handmatig tunen of uit odometry halen

        dx = scale * t[0][0]
        dy = scale * t[1][0]

        # Update pose
        self.x += dx
        self.y += dy
        self.theta += d_theta

        # Publish
        pose_msg = Pose2D()
        pose_msg.x = self.x
        pose_msg.y = self.y
        pose_msg.theta = self.theta

        self.pose_pub.publish(pose_msg)

        # Update
        self.prev_gray = gray
        self.prev_pts = good_next.reshape(-1, 1, 2)


if __name__ == "__main__":
    node = SLAMNode()
    rospy.spin()