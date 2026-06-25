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
        self.node_positions = self.load_map(self.map_file)

        self.path = []
        self.current_node = None

        rospy.Subscriber("/odometry", Odometry, self.odom_callback)
        rospy.Subscriber("/planned_path", String, self.path_callback)

        self.pub_current = rospy.Publisher("/current_node", String, queue_size=1)
        self.pub_next = rospy.Publisher("/next_node", String, queue_size=1)
        self.pub_remaining = rospy.Publisher("/remaining_path", String, queue_size=1)

        rospy.loginfo("=== LOCALIZATION NODE STARTED ===")
        rospy.loginfo(f"Loaded {len(self.node_positions)} nodes from map.")

    def load_map(self, map_file):
        with open(map_file, 'r') as file:
            data = yaml.safe_load(file)

        node_positions = {}
        for node, coords in data['nodes'].items():
            node_positions[node] = (float(coords[0]), float(coords[1]))

        return node_positions

    def path_callback(self, msg):
        raw = msg.data.replace(" ", "")

        if "->" in raw:
            self.path = raw.split("->")
        else:
            self.path = raw.split(",")

        rospy.loginfo(f"📍 Received path: {self.path}")

    def get_closest_node(self, x, y):
        closest_node = None
        min_dist = float('inf')

        for node, (nx, ny) in self.node_positions.items():
            dist = math.sqrt((x - nx)**2 + (y - ny)**2)

            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node

    def odom_callback(self, msg):

        if not self.path:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        detected_node = self.get_closest_node(x, y)

        if detected_node != self.current_node:

            self.current_node = detected_node

            rospy.loginfo("===================================")
            rospy.loginfo(f"🤖 Robot pose: x={x:.2f}, y={y:.2f}")
            rospy.loginfo(f"📍 Current node: {self.current_node}")

            if self.current_node in self.path:
                index = self.path.index(self.current_node)

                next_node = None
                if index + 1 < len(self.path):
                    next_node = self.path[index + 1]

                remaining = self.path[index:]

                rospy.loginfo(f"➡ Next node: {next_node}")
                rospy.loginfo(f"🛣 Remaining path: {remaining}")

                self.pub_current.publish(self.current_node)
                self.pub_next.publish(next_node if next_node else "GOAL_REACHED")
                self.pub_remaining.publish("->".join(remaining))


if __name__ == '__main__':
    try:
        node = CalibratedLocalization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass