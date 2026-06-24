#!/bin/bash

source /environment.sh
dt-launchfile-init

# Pad naar YAML kaartbestand
MAP_FILE="/code/catkin_ws/src/timros1/packages/my_package/src/map.yaml"

echo "Using map file: $MAP_FILE"

# 1. Odometry
rosrun my_package odometry.py &
sleep 2
echo "[slam-launch] ✓ odometry gestart"

# Start Dijkstra (leest start/goal uit YAML)
rosrun my_package yaml_dijkstra_node.py \
    _map_file:=$MAP_FILE &

# Start Localization
rosrun my_package calibrated_localization.py \
    _map_file:=$MAP_FILE &

dt-launchfile-join
