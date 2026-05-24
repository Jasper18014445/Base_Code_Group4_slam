#!/usr/bin/env python3
"""
slam_viz.py  –  SLAM kaartvisualisatie voor Duckietown
=======================================================
Aanpak: de node schrijft de kaart elke seconde als PNG naar
/data/slam_map.png en /data/slam_map.json.

Die bestanden zijn bereikbaar via de Duckietown bestandsbrowser op
  http://<robot-ip>:8082/files/data/slam_map.png

De meegeleverde slam_viewer.html leest /data/slam_map.json via de
Duckietown HTTP-server en toont de kaart in je browser.
Geen extra poorten nodig.

Publiceert ook op /slam_map (OccupancyGrid) voor RViz.
"""

import rospy
import json
import os
import math
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose2D

OUTPUT_DIR  = "/data"
PNG_PATH    = os.path.join(OUTPUT_DIR, "slam_map.png")
JSON_PATH   = os.path.join(OUTPUT_DIR, "slam_map.json")
UPDATE_HZ   = 2.0   # hoe vaak per seconde opslaan

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Kleurenkaart ─────────────────────────────────────────────────────────────
#   -1 / 255 = onbekend  →  #0a0e17 donkerblauw
#    0       = vrij       →  #0d1f2d blauwgrijs
#   50       = robot      →  #ff6b35 oranje
#  100       = bezet      →  #00d4ff cyaan

def grid_to_rgb(val: int) -> tuple:
    if val < 0 or val == 255:   return (23, 14, 10)   # onbekend (BGR)
    if val == 0:                return (45, 31, 13)    # vrij
    if val == 50:               return (53, 107, 255)  # robot
    t = val / 100.0
    b = int(255 * t)
    g = int(212 * t)
    r = 0
    return (b, g, r)                                   # bezet


class SLAMVizNode:
    def __init__(self):
        rospy.init_node("slam_viz_node")

        self._map_msg  = None
        self._pose     = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self._traj     = []
        self._landmarks = 0

        rospy.Subscriber("/slam_map",    OccupancyGrid, self._map_cb,  queue_size=1)
        rospy.Subscriber("/fused_pose",  Pose2D,        self._pose_cb, queue_size=5)
        rospy.Subscriber("/visual_pose", Pose2D,        self._pose_cb, queue_size=5)

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / UPDATE_HZ), self._save_tick
        )

        rospy.loginfo("[slam_viz] Gestart — kaart wordt opgeslagen naar %s", OUTPUT_DIR)
        rospy.loginfo("[slam_viz] Open slam_viewer.html in je browser op je laptop.")

    def _map_cb(self, msg: OccupancyGrid):
        self._map_msg = msg
        self._landmarks = sum(1 for v in msg.data if v >= 100)

    def _pose_cb(self, msg: Pose2D):
        self._pose = {"x": msg.x, "y": msg.y, "theta": msg.theta}
        self._traj.append((msg.x, msg.y))
        if len(self._traj) > 3000:
            self._traj = self._traj[-3000:]

    def _save_tick(self, _event):
        if self._map_msg is None:
            return
        msg = self._map_msg
        W, H = msg.info.width, msg.info.height
        res  = msg.info.resolution

        # ── Bouw RGB-afbeelding ──────────────────────────────────────────────
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for row in range(H):
            for col in range(W):
                idx = (H - 1 - row) * W + col   # flip Y
                val = msg.data[idx] if idx < len(msg.data) else -1
                canvas[row, col] = grid_to_rgb(int(val))

        # Teken het traject
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        for i in range(1, len(self._traj)):
            x0, y0 = self._traj[i-1]
            x1, y1 = self._traj[i]
            c0 = int((x0 - ox) / res), H - 1 - int((y0 - oy) / res)
            c1 = int((x1 - ox) / res), H - 1 - int((y1 - oy) / res)
            if all(0 <= v < d for v, d in zip(c0 + c1, [W, H, W, H])):
                cv2.line(canvas, c0, c1, (64, 255, 0), 2)

        # Teken de robot
        rx = int((self._pose["x"] - ox) / res)
        ry = H - 1 - int((self._pose["y"] - oy) / res)
        if 0 <= rx < W and 0 <= ry < H:
            theta = -self._pose["theta"]
            tip   = (int(rx + 12 * math.cos(theta)),
                     int(ry + 12 * math.sin(theta)))
            l     = (int(rx -  6 * math.cos(theta) + 6 * math.sin(theta)),
                     int(ry -  6 * math.sin(theta) - 6 * math.cos(theta)))
            r_pt  = (int(rx -  6 * math.cos(theta) - 6 * math.sin(theta)),
                     int(ry -  6 * math.sin(theta) + 6 * math.cos(theta)))
            cv2.fillPoly(canvas, [np.array([tip, l, r_pt])], (53, 107, 255))
            cv2.polylines(canvas, [np.array([tip, l, r_pt])], True, (255, 255, 255), 1)

        # Schaal de afbeelding op zodat kleine kaarten goed zichtbaar zijn
        scale = max(1, min(4, 800 // max(W, H)))
        if scale > 1:
            canvas = cv2.resize(canvas, (W * scale, H * scale),
                                interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(PNG_PATH, canvas)

        # ── JSON metadata ────────────────────────────────────────────────────
        meta = {
            "ts":          rospy.Time.now().to_sec(),
            "width":       W,
            "height":      H,
            "resolution":  res,
            "origin_x":    ox,
            "origin_y":    oy,
            "robot":       self._pose,
            "landmarks":   self._landmarks,
            "traj_len":    len(self._traj),
            "png":         "slam_map.png",
        }
        with open(JSON_PATH, "w") as f:
            json.dump(meta, f)

        rospy.loginfo_throttle(5.0,
            "[slam_viz] Kaart opgeslagen: %dx%d cellen, %d features, robot=(%.2f, %.2f)",
            W, H, self._landmarks, self._pose["x"], self._pose["y"])


if __name__ == "__main__":
    node = SLAMVizNode()
    rospy.spin()
