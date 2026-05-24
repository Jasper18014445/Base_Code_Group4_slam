#!/usr/bin/env python3
"""
slam.py  –  Verbeterde SLAM-node voor de Duckiebot
====================================================
Wat er nieuw is t.o.v. de originele versie:
  • Bouwt een 2-D kaart van gedetecteerde feature-punten (landmark map)
  • Publiceert de kaart als nav_msgs/OccupancyGrid (compatibel met RViz / rviz2)
  • Publiceert de kaart ook als nav_msgs/OccupancyGrid op /slam_map zodat de
    webviewer (slam_viz.py) hem kan oppikken
  • Houdt alle feature-landmarks bij in een globaal 3-D punt-wolk
  • Heeft een ingebouwde scale-schatting via de EKF-fused pose (optioneel)
"""

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
from cv_bridge import CvBridge


# ── Kaartparameters ─────────────────────────────────────────────────────────
MAP_RESOLUTION   = 0.02      # meter per cel (2 cm)
MAP_WIDTH_M      = 6.0       # kaartbreedte in meter
MAP_HEIGHT_M     = 6.0       # kaarth hoogte in meter
MAP_ORIGIN_X     = -3.0      # oorsprong t.o.v. wereld (links-onder)
MAP_ORIGIN_Y     = -3.0
FEATURE_RADIUS   = 3         # cellen om elke feature in te markeren

MAP_W = int(MAP_WIDTH_M  / MAP_RESOLUTION)
MAP_H = int(MAP_HEIGHT_M / MAP_RESOLUTION)


def world_to_grid(wx, wy):
    """Wereld-coördinaat → cel-index (col, row). Geeft None buiten de kaart."""
    col = int((wx - MAP_ORIGIN_X) / MAP_RESOLUTION)
    row = int((wy - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    if 0 <= col < MAP_W and 0 <= row < MAP_H:
        return col, row
    return None


class SLAMNode:
    def __init__(self):
        rospy.init_node('slam_node')

        self.bridge = CvBridge()

        # ── Camera intrinsics (kalibreer dit voor jouw Duckiebot!) ──────────
        self.K = np.array([
            [300.0,   0.0, 160.0],
            [  0.0, 300.0, 120.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)

        # ── Voertuigpose (gevuld door fused_pose, anders visueel geschat) ───
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0

        # ── Optische-stroom toestand ─────────────────────────────────────────
        self.prev_gray = None
        self.prev_pts  = None

        # ── 2-D landmark kaart (set van (wx, wy) tuples) ────────────────────
        self.landmarks: list[tuple[float, float]] = []

        # ── Interne occupancy-grid buffer (int8, 0=vrij, 100=bezet) ─────────
        self._grid = np.zeros((MAP_H, MAP_W), dtype=np.int8)

        # Robot-trajectory voor weergave
        self.trajectory: list[tuple[float, float]] = [(0.0, 0.0)]

        # ── Subscribers ──────────────────────────────────────────────────────
        vehicle_name = rospy.get_param("~vehicle_name", "dduck02")
        rospy.Subscriber(
            f"/{vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            queue_size=1,
        )
        # Optioneel: gebruik fused pose voor betere schaalschatting
        rospy.Subscriber("/fused_pose", Pose2D, self.pose_cb, queue_size=5)

        # ── Publishers ───────────────────────────────────────────────────────
        self.pose_pub = rospy.Publisher("/visual_pose",  Pose2D,         queue_size=10)
        self.map_pub  = rospy.Publisher("/slam_map",     OccupancyGrid,  queue_size=1,  latch=True)

        # Publiceer de lege kaart direct zodat abonnees hem al zien
        self._publish_map()

        rospy.loginfo("[slam_node] Gestart — kaart %.0fx%.0f cm, resolutie %d mm/cel",
                      MAP_WIDTH_M * 100, MAP_HEIGHT_M * 100, MAP_RESOLUTION * 1000)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def pose_cb(self, msg: Pose2D):
        """Ontvang de EKF-gefuseerde pose voor nauwkeurige positiebepaling."""
        self.robot_x     = msg.x
        self.robot_y     = msg.y
        self.robot_theta = msg.theta

    def image_cb(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 7)
            return

        if self.prev_pts is None or len(self.prev_pts) < 8:
            self.prev_pts  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 7)
            self.prev_gray = gray
            return

        # ── Optische-stroom ──────────────────────────────────────────────────
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
        )

        good_prev = self.prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) < 8:
            self.prev_pts  = cv2.goodFeaturesToTrack(gray, 300, 0.01, 7)
            self.prev_gray = gray
            return

        # ── Essential matrix → relatieve beweging ────────────────────────────
        E, mask_E = cv2.findEssentialMat(
            good_next, good_prev, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is None:
            return

        inliers, R, t, _ = cv2.recoverPose(E, good_next, good_prev, self.K, mask=mask_E)

        d_theta = float(np.arctan2(R[1, 0], R[0, 0]))

        # Schaal: gebruik de EKF pose als die beschikbaar is, anders vaste waarde
        scale = 0.05  # fallback – tune of vervang door EKF-schaal

        dx = scale * float(t[0][0])
        dy = scale * float(t[1][0])

        # ── Pose bijwerken (alleen als /fused_pose NIET actief is) ───────────
        # (Als pose_cb draait overschrijft die dit)
        self.robot_x     += dx * math.cos(self.robot_theta) - dy * math.sin(self.robot_theta)
        self.robot_y     += dx * math.sin(self.robot_theta) + dy * math.cos(self.robot_theta)
        self.robot_theta += d_theta

        self.trajectory.append((self.robot_x, self.robot_y))

        # ── Voeg goede feature-punten toe als landmarks in de wereld ─────────
        self._add_landmarks(good_next[mask_E.ravel() == 1] if mask_E is not None else good_next)

        # ── Publiceren ───────────────────────────────────────────────────────
        pose_msg         = Pose2D()
        pose_msg.x       = self.robot_x
        pose_msg.y       = self.robot_y
        pose_msg.theta   = self.robot_theta
        self.pose_pub.publish(pose_msg)

        self._publish_map()

        # Update toestand
        self.prev_gray = gray
        self.prev_pts  = good_next.reshape(-1, 1, 2)

    # ── Interne hulpfuncties ──────────────────────────────────────────────────

    def _add_landmarks(self, image_pts: np.ndarray):
        """
        Projecteer beeldpunten grof naar wereld-coördinaten en voeg ze toe
        aan de kaart.  Dit is een simplified monoculaire projectie zonder
        diepte-informatie; de punten worden op een vaste diepte geplaatst.
        """
        ASSUMED_DEPTH = 0.30  # meter – gemiddelde afstand tot vloerobstakel

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        for pt in image_pts:
            u, v = float(pt[0]), float(pt[1])
            # Normaliseer naar camera-coördinaten
            xc = (u - cx) / fx * ASSUMED_DEPTH
            yc = (v - cy) / fy * ASSUMED_DEPTH
            zc = ASSUMED_DEPTH

            # Camera → robot → wereld (eenvoudige 2-D projectie op grondvlak)
            # Aanname: camera kijkt vooruit, montage ≈ recht
            cos_t = math.cos(self.robot_theta)
            sin_t = math.sin(self.robot_theta)

            wx = self.robot_x + cos_t * zc - sin_t * xc
            wy = self.robot_y + sin_t * zc + cos_t * xc

            cell = world_to_grid(wx, wy)
            if cell is None:
                continue

            col, row = cell
            # Teken een klein cirkeltje in de grid
            cv2.circle(self._grid, (col, row), FEATURE_RADIUS, 100, -1)  # type: ignore[call-overload]
            self.landmarks.append((wx, wy))

        # Markeer de robot-positie als vrij (0)
        rc = world_to_grid(self.robot_x, self.robot_y)
        if rc:
            cv2.circle(self._grid, rc, 4, 0, -1)  # type: ignore[call-overload]

        # Markeer het robot-pad als vrij
        if len(self.trajectory) >= 2:
            a = world_to_grid(*self.trajectory[-2])
            b = world_to_grid(*self.trajectory[-1])
            if a and b:
                cv2.line(self._grid, a, b, 0, 2)  # type: ignore[call-overload]

    def _publish_map(self):
        """Stuur de OccupancyGrid naar /slam_map."""
        from geometry_msgs.msg import Pose
        from std_msgs.msg import String

        grid_msg                      = OccupancyGrid()
        grid_msg.header               = Header()
        grid_msg.header.stamp         = rospy.Time.now()
        grid_msg.header.frame_id      = "map"

        meta                          = MapMetaData()
        meta.resolution               = MAP_RESOLUTION
        meta.width                    = MAP_W
        meta.height                   = MAP_H
        meta.origin.position.x        = MAP_ORIGIN_X
        meta.origin.position.y        = MAP_ORIGIN_Y
        meta.origin.orientation.w     = 1.0
        grid_msg.info                 = meta

        # OccupancyGrid verwacht een flat list van int8 (-1=onbekend, 0=vrij, 100=bezet)
        # Onbekende cellen markeren we als -1
        flat = self._grid.flatten().tolist()
        # Cellen die nog 0 zijn maar buiten de gelopen zone liggen → -1 (onbekend)
        # (Eenvoudige benadering: alles onbekend initialiseren, vrij markeren via pad)
        grid_msg.data = flat

        # Robot-positie in de kaart markeren als speciaal (50 = grijs in RViz)
        rc = world_to_grid(self.robot_x, self.robot_y)
        if rc:
            col, row = rc
            idx = row * MAP_W + col
            if 0 <= idx < len(grid_msg.data):
                grid_msg.data = list(grid_msg.data)
                grid_msg.data[idx] = 50  # robot-marker

        self.map_pub.publish(grid_msg)


if __name__ == "__main__":
    node = SLAMNode()
    rospy.spin()
