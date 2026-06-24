#!/usr/bin/env python3
"""
slam.py  Monoculaire SLAM voor Duckiebot

"""

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from std_msgs.msg import Header

# ── Kaartinstellingen ─────────────────────────────────────────────────────────
MAP_RES    = 0.025        # meter per cel (2.5 cm)
MAP_SIZE_M = 5.0          # kaart is 5x5 meter
MAP_N      = int(MAP_SIZE_M / MAP_RES)   # aantal cellen per kant (200x200)
MAP_ORIGIN = -MAP_SIZE_M / 2.0           # oorsprong in meters (-2.5)

# Aangenomen camera-hoogte boven de grond (voor projectie naar vloer)
CAM_HEIGHT = 0.10         # meter


class SLAMNode:
    def __init__(self):
        rospy.init_node('slam_node')

        vehicle_name = rospy.get_param("~vehicle_name", "dduck02")

        # ── Camera intrinsics ─────────────────────────────────────────────────
        # fx, fy = brandpuntsafstand in pixels
        # cx, cy = beeldmiddelpunt
        # Pas dit aan na camerakalibratie!
        self.K  = np.array([[300., 0., 160.],
                             [0., 300., 120.],
                             [0.,   0.,   1.]], dtype=np.float64)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        # ── Robot-pose (ingevuld door odometry) ───────────────────────────────
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0
        self.odom_speed  = 0.0   # m/s uit odometry, voor schaalschatting

        # ── Visuele pose (geaccumuleerd uit frame-to-frame beweging) ──────────
        self.vis_x     = 0.0
        self.vis_y     = 0.0
        self.vis_theta = 0.0

        # ── Optical flow toestand ─────────────────────────────────────────────
        self.prev_gray    = None
        self.prev_pts     = None
        self.prev_stamp   = None

        # ── Landmark opslag ───────────────────────────────────────────────────
        # Elke landmark is een (wx, wy) tuple in meter, wereldcoördinaten
        self.landmarks: list = []

        # ── Occupancy grid buffer ─────────────────────────────────────────────
        # -1 = onbekend, 0 = vrij, 100 = bezet
        self._grid = np.full((MAP_N, MAP_N), -1, dtype=np.int8)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pose_pub = rospy.Publisher("/visual_pose", Pose2D,
                                        queue_size=10)
        self.map_pub  = rospy.Publisher("/slam_map", OccupancyGrid,
                                        queue_size=1, latch=True)

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber(
            f"/{vehicle_name}/camera_node/image/compressed",
            CompressedImage, self._image_cb, queue_size=1)

        rospy.Subscriber(
            "/odometry", Odometry, self._odom_cb, queue_size=5)

        # Stuur direct een lege kaart zodat de viewer meteen iets ziet
        self._publish_map()
        rospy.loginfo("[slam] Gestart — kaart %dx%d, res=%.0f mm/cel",
                      MAP_N, MAP_N, MAP_RES * 1000)

    # ── Odometry callback: robot-pose + snelheid bijhouden ───────────────────

    def _odom_cb(self, msg: Odometry):
        self.robot_x     = msg.pose.pose.position.x
        self.robot_y     = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_theta = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))
        self.odom_speed  = msg.twist.twist.linear.x

    # ── Beeld callback ────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        # Decoderen
        arr   = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stamp = msg.header.stamp

        if self.prev_gray is None:
            self._reset_features(gray, stamp)
            return

        # ── Optical flow: track bestaande punten ─────────────────────────────
        if self.prev_pts is None or len(self.prev_pts) < 8:
            self._reset_features(gray, stamp)
            return

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3)

        ok        = status.ravel() == 1
        good_prev = self.prev_pts[ok]
        good_next = next_pts[ok]

        if len(good_prev) < 8:
            self._reset_features(gray, stamp)
            return

        # ── Essential matrix → relatieve rotatie + translatie ─────────────────
        E, mask_E = cv2.findEssentialMat(
            good_next, good_prev, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            self._reset_features(gray, stamp)
            return

        inliers, R, t, _ = cv2.recoverPose(
            E, good_next, good_prev, self.K, mask=mask_E)
        if inliers < 8:
            return

        # ── Schaal schatten via odometry ──────────────────────────────────────
        dt = (stamp - self.prev_stamp).to_sec() if self.prev_stamp else 0.1
        dt = max(dt, 0.01)

        # Gebruik de werkelijke odometry-snelheid × dt als schaalfactor.
        # Als de robot stilstaat, kleine schaal gebruiken om drift te beperken.
        speed  = abs(self.odom_speed)
        scale  = speed * dt if speed > 0.01 else 0.001

        # ── Visuele pose accumuleren ──────────────────────────────────────────
        d_theta = float(np.arctan2(R[1, 0], R[0, 0]))
        dx_cam  = scale * float(t[0][0])
        dy_cam  = scale * float(t[1][0])

        # Camera-beweging → wereld-beweging (roteer met huidige oriëntatie)
        cos_t = math.cos(self.vis_theta)
        sin_t = math.sin(self.vis_theta)
        self.vis_x     += cos_t * dx_cam - sin_t * dy_cam
        self.vis_y     += sin_t * dx_cam + cos_t * dy_cam
        self.vis_theta += d_theta

        # ── Inlier feature-punten → landmarks op de kaart zetten ─────────────
        inlier_mask = mask_E.ravel() == 255 if mask_E is not None else np.ones(len(good_next), bool)
        inlier_pts  = good_next[inlier_mask]

        # Gebruik de ROBOT-pose (nauwkeuriger dan vis_pose) als referentie
        rx, ry, rt = self.robot_x, self.robot_y, self.robot_theta

        new_landmarks = []
        for pt in inlier_pts:
            u, v = float(pt[0]), float(pt[1])

            # Alleen punten in de onderste helft van het beeld zijn
            # waarschijnlijk grond/obstakels dichtbij de robot
            if v < frame.shape[0] * 0.4:
                continue

            # Projecteer beeldpunt naar grondvlak via perspectief
            # Aanname: camera kijkt horizontaal vooruit op hoogte CAM_HEIGHT
            # Punten lager in beeld = dichterbij
            # Diepte-schatting: d = f * h / (v - cy) voor v > cy
            if v <= self.cy + 5:
                continue

            depth = self.fy * CAM_HEIGHT / (v - self.cy)
            depth = min(max(depth, 0.05), 2.0)   # clamp: 5 cm – 2 m

            # Camera-coördinaten
            xc = (u - self.cx) / self.fx * depth
            zc = depth   # voorwaarts

            # Robot-coördinaten → wereld-coördinaten
            cos_r = math.cos(rt)
            sin_r = math.sin(rt)
            wx = rx + cos_r * zc - sin_r * xc
            wy = ry + sin_r * zc + cos_r * xc

            # Naar cel-index
            col = int((wx - MAP_ORIGIN) / MAP_RES)
            row = int((wy - MAP_ORIGIN) / MAP_RES)

            if 0 <= col < MAP_N and 0 <= row < MAP_N:
                # Flip Y voor weergave (rij 0 = bovenkant kaart = hoge Y)
                draw_row = MAP_N - 1 - row
                # Markeer als bezet
                self._grid[draw_row, col] = 100
                new_landmarks.append((wx, wy))

        # Robot-positie + pad als "vrij" markeren
        self._mark_free(rx, ry)

        if new_landmarks:
            self.landmarks.extend(new_landmarks)
            # Begrens geheugen
            if len(self.landmarks) > 50000:
                self.landmarks = self.landmarks[-50000:]

        # ── Publiceren ────────────────────────────────────────────────────────
        pose_msg       = Pose2D()
        pose_msg.x     = self.vis_x
        pose_msg.y     = self.vis_y
        pose_msg.theta = self.vis_theta
        self.pose_pub.publish(pose_msg)

        self._publish_map()

        # ── Toestand updaten ──────────────────────────────────────────────────
        self.prev_gray  = gray
        self.prev_stamp = stamp
        # Gebruik ALLE getrackte punten (niet alleen inliers) als startpunten
        # voor het volgende frame, aangevuld met nieuwe features
        self.prev_pts = good_next.reshape(-1, 1, 2)

        # Periodiek nieuwe features toevoegen als er te weinig zijn
        if len(self.prev_pts) < 50:
            new_pts = cv2.goodFeaturesToTrack(
                gray, 200, 0.01, 7,
                mask=self._feature_mask(frame.shape))
            if new_pts is not None:
                self.prev_pts = np.vstack([self.prev_pts, new_pts])

        rospy.loginfo_throttle(2.0,
            "[slam] %d landmarks | vis_pos=(%.2f, %.2f) | "
            "robot=(%.2f, %.2f) | scale=%.4f",
            len(self.landmarks),
            self.vis_x, self.vis_y,
            self.robot_x, self.robot_y,
            scale)

    # ── Hulpfuncties ──────────────────────────────────────────────────────────

    def _reset_features(self, gray, stamp):
        """Herdetecteer features na verlies."""
        h = gray.shape[0]
        # Alleen features in onderste 60% van beeld (vloer/obstakels)
        mask = np.zeros_like(gray)
        mask[int(h * 0.4):, :] = 255
        self.prev_pts   = cv2.goodFeaturesToTrack(
            gray, 300, 0.01, 7, mask=mask)
        self.prev_gray  = gray
        self.prev_stamp = stamp

    def _feature_mask(self, shape):
        """Masker: alleen onderste 60% van beeld."""
        mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        mask[int(shape[0] * 0.4):, :] = 255
        return mask

    def _mark_free(self, wx, wy):
        """Markeer robot-positie als vrij op de kaart."""
        col = int((wx - MAP_ORIGIN) / MAP_RES)
        row = MAP_N - 1 - int((wy - MAP_ORIGIN) / MAP_RES)
        if 0 <= col < MAP_N and 0 <= row < MAP_N:
            # Klein cirkeltje vrijmaken
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    r2, c2 = row + dr, col + dc
                    if 0 <= r2 < MAP_N and 0 <= c2 < MAP_N:
                        if self._grid[r2, c2] != 100:
                            self._grid[r2, c2] = 0

    def _publish_map(self):
        """Stuur de OccupancyGrid naar /slam_map."""
        msg                          = OccupancyGrid()
        msg.header                   = Header()
        msg.header.stamp             = rospy.Time.now()
        msg.header.frame_id          = "map"
        msg.info                     = MapMetaData()
        msg.info.resolution          = MAP_RES
        msg.info.width               = MAP_N
        msg.info.height              = MAP_N
        msg.info.origin.position.x   = MAP_ORIGIN
        msg.info.origin.position.y   = MAP_ORIGIN
        msg.info.origin.orientation.w = 1.0
        msg.data                     = self._grid.flatten().tolist()
        self.map_pub.publish(msg)


if __name__ == "__main__":
    node = SLAMNode()
    rospy.spin()
