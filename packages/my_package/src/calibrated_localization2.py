#!/usr/bin/env python3

import rospy
import yaml
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class CalibratedLocalization:

    def __init__(self):
        rospy.init_node('calibrated_localization_node')

        self.map_file = rospy.get_param('~map_file')
        self.max_node_dist = rospy.get_param('~max_node_dist', 0.30)  # meter

        self.node_positions = self.load_map(self.map_file)

        self.path = []
        self.current_node = None

        # Debug counters
        self.odom_msg_count = 0
        self.path_msg_count = 0
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_closest = None
        self.last_closest_dist = float('inf')

        rospy.Subscriber("odometry", Odometry, self.odom_callback)
        rospy.Subscriber("planned_path", String, self.path_callback)

        self.pub_current = rospy.Publisher("current_node", String, queue_size=1)
        self.pub_next = rospy.Publisher("next_node", String, queue_size=1)
        self.pub_remaining = rospy.Publisher("remaining_path", String, queue_size=1)

        rospy.loginfo("=" * 60)
        rospy.loginfo("LOCALIZATION NODE STARTED")
        rospy.loginfo(f"  map_file       = {self.map_file}")
        rospy.loginfo(f"  max_node_dist  = {self.max_node_dist} m")
        rospy.loginfo(f"  loaded nodes   = {len(self.node_positions)}")
        for n, (nx, ny) in self.node_positions.items():
            rospy.loginfo(f"    {n}: ({nx:.3f}, {ny:.3f})")
        rospy.loginfo("=" * 60)

        # Heartbeat-timer: vertelt je elke 2s of er überhaupt iets binnenkomt
        rospy.Timer(rospy.Duration(2.0), self.heartbeat_cb)

    # ------------------------------------------------------------------
    def load_map(self, map_file):
        with open(map_file, 'r') as file:
            data = yaml.safe_load(file)

        node_positions = {}
        for node, coords in data['nodes'].items():
            node_positions[node] = (float(coords[0]), float(coords[1]))
        return node_positions

    # ------------------------------------------------------------------
    def heartbeat_cb(self, _evt):
        rospy.loginfo(
            f"[HEARTBEAT] odom_msgs={self.odom_msg_count} "
            f"path_msgs={self.path_msg_count} "
            f"last_pose=({self.last_x:.3f}, {self.last_y:.3f}) "
            f"closest={self.last_closest} (d={self.last_closest_dist:.3f}m) "
            f"current_node={self.current_node} "
            f"path_len={len(self.path)}"
        )

        if self.odom_msg_count == 0:
            rospy.logwarn("[HEARTBEAT] Geen ODOMETRY berichten ontvangen! "
                          "Check topic-naam/namespace van /odometry.")
        if self.path_msg_count == 0:
            rospy.logwarn("[HEARTBEAT] Geen PLANNED_PATH ontvangen — "
                          "odom_callback returnt vroeg, geen node-detectie.")

    # ------------------------------------------------------------------
    def path_callback(self, msg):
        self.path_msg_count += 1
        raw = msg.data.replace(" ", "")

        if "->" in raw:
            self.path = raw.split("->")
        else:
            self.path = raw.split(",")

        rospy.loginfo(f"📍 Received path ({len(self.path)} nodes): {self.path}")

        # Check of alle nodes in het pad ook in de map staan
        unknown = [n for n in self.path if n not in self.node_positions]
        if unknown:
            rospy.logerr(f"⚠️ Path bevat onbekende nodes: {unknown}")

    # ------------------------------------------------------------------
    def get_closest_node(self, x, y):
        closest_node = None
        min_dist = float('inf')

        for node, (nx, ny) in self.node_positions.items():
            dist = math.hypot(x - nx, y - ny)
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node, min_dist

    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.odom_msg_count += 1

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_x = x
        self.last_y = y

        # Per-message debug (throttled)
        rospy.loginfo_throttle(
            1.0,
            f"[ODOM] x={x:.3f} y={y:.3f} "
            f"frame_id={msg.header.frame_id}"
        )

        if not self.path:
            rospy.loginfo_throttle(
                2.0, "⏳ Odom binnen, maar nog geen planned_path ontvangen."
            )
            return

        closest, dist = self.get_closest_node(x, y)
        self.last_closest = closest
        self.last_closest_dist = dist

        # Drempel: alleen accepteren als je dicht genoeg bij een node bent
        if dist > self.max_node_dist:
            rospy.loginfo_throttle(
                1.0,
                f"[DETECT] dichtstbijzijnde = {closest} "
                f"maar {dist:.3f}m > {self.max_node_dist}m → genegeerd"
            )
            return

        detected_node = closest

        if detected_node != self.current_node:
            self.current_node = detected_node

            rospy.loginfo("===================================")
            rospy.loginfo(f"🤖 Robot pose: x={x:.3f}, y={y:.3f}")
            rospy.loginfo(f"📍 Current node: {self.current_node} "
                          f"(afstand: {dist:.3f}m)")

            if self.current_node in self.path:
                index = self.path.index(self.current_node)

                next_node = None
                if index + 1 < len(self.path):
                    next_node = self.path[index + 1]

                remaining = self.path[index:]

                rospy.loginfo(f"➡ Next node: {next_node}")
                rospy.loginfo(f"🛣 Remaining path: {remaining}")

                self.pub_current.publish(str(self.current_node))
                self.pub_next.publish(str(next_node) if next_node else "GOAL_REACHED")
                self.pub_remaining.publish("->".join(remaining))
            else:
                rospy.logwarn(
                    f"⚠️ Current node '{self.current_node}' "
                    f"zit NIET in het geplande pad {self.path} "
                    f"→ niets gepubliceerd."
                )


if __name__ == '__main__':
    try:
        node = CalibratedLocalization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
