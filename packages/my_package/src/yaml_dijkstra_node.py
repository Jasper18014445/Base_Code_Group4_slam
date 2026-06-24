#!/usr/bin/env python3

import rospy
import yaml
import heapq
from std_msgs.msg import String

rospy.loginfo("DIJKSTRA INIT CALLED")


class YAMLDijkstraNode:

    def __init__(self):
        rospy.init_node('yaml_dijkstra_node')

        # Parameter: map file is required
        map_file = rospy.get_param('~map_file')

        # Load full YAML once
        with open(map_file, 'r') as file:
            self.map_data = yaml.safe_load(file)

        # ✅ Read start/goal directly from YAML
        try:
            self.start_node = self.map_data['start_node']
            self.goal_node = self.map_data['goal_node']
        except KeyError:
            rospy.logerr("Map YAML must contain 'start_node' and 'goal_node'.")
            rospy.signal_shutdown("Missing start/goal in YAML")
            return

        rospy.loginfo("=== DIJKSTRA NODE STARTED ===")
        rospy.loginfo(f"Map file: {map_file}")
        rospy.loginfo(f"Start node (YAML): {self.start_node}")
        rospy.loginfo(f"Goal node (YAML): {self.goal_node}")

        # Publisher
        self.pub_path = rospy.Publisher('/planned_path', String, queue_size=10, latch=True)

        # Load graph from YAML
        self.graph = self.load_graph(self.map_data)

        # Validate nodes exist
        if self.start_node not in self.graph or self.goal_node not in self.graph:
            rospy.logerr("Start or goal node not found in graph edges.")
            rospy.signal_shutdown("Invalid start/goal node")
            return

        # Compute path
        path = self.dijkstra(self.graph, self.start_node, self.goal_node)

        if not path:
            rospy.logerr("No path found between start and goal.")
            rospy.signal_shutdown("Path planning failed")
            return

        path_string = "->".join(path)

        rospy.loginfo(f"✅ Planned path: {path_string}")
        rospy.loginfo(f"✅ Path length: {len(path)} nodes")

        self.pub_path.publish(path_string)

    def load_graph(self, data):
        graph = {}

        if 'edges' not in data:
            rospy.logerr("Map YAML must contain 'edges' section.")
            rospy.signal_shutdown("Invalid YAML structure")
            return {}

        for node in data['edges']:
            graph[node] = []
            for neighbor, weight in data['edges'][node].items():
                graph[node].append((neighbor, weight))

        rospy.loginfo(f"Loaded {len(graph)} nodes from map.")
        return graph

    def dijkstra(self, graph, start, goal):
        pq = [(0, start)]
        dist = {node: float('inf') for node in graph}
        previous = {node: None for node in graph}

        dist[start] = 0

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node == goal:
                break

            if current_dist > dist[current_node]:
                continue

            for neighbor, weight in graph[current_node]:
                distance = current_dist + weight

                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        # Reconstruct path
        path = []
        node = goal

        while node is not None:
            path.append(node)
            node = previous[node]

        path.reverse()

        # If start not reachable
        if path[0] != start:
            return []

        return path


if __name__ == '__main__':
    try:
        YAMLDijkstraNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
