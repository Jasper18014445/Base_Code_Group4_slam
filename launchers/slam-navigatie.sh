#!/bin/bash
source /environment.sh
dt-launchfile-init

# Pad naar YAML kaartbestand (zelfde als calibrated_localization.sh)
MAP_FILE="/code/catkin_ws/src/timros1/packages/my_package/src/map.yaml"
echo "Using map file: $MAP_FILE"

# 1. Odometry — levert /odometry (positie uit wielencoders)
rosrun my_package odometry.py &
sleep 2
echo "[slam-navigatie] odometry gestart"

# 2. Taak 1 — Padplanning: bereken route, publiceer op /planned_path
rosrun my_package yaml_dijkstra_node.py \
    _map_file:=$MAP_FILE &
sleep 2
echo "[slam-navigatie] dijkstra gestart"

# 3. Taak 2 — Lokalisatie: huidige/volgende node uit positie + route
rosrun my_package calibrated_localization.py \
    _map_file:=$MAP_FILE &
sleep 2
echo "[slam-navigatie] lokalisatie gestart"

# 4. Duckie-detector — publiceert /<vehicle>/duckie_stop bij een eendje
#rosrun my_package duckie_detector_node.py &  #tijdelijk uitgezet
#sleep 2
echo "[slam-navigatie] duckie-detector gestart"

# 5. Taak 4 — Lane-following met PID (rijdt lijnen, pauzeert bij kruispunt)
rosrun my_package lane_following_duckie.py &
sleep 2
echo "[slam-navigatie] lane-following (PID) gestart"

# 6. Taak 3 — Navigatie: rode stoplijn detecteren, draaien volgens route
rosrun my_package intersection_navigator.py \
    _map_file:=$MAP_FILE &
sleep 1
echo "[slam-navigatie] navigatie gestart"

echo ""
echo "=========================================================="
echo "  Alle nodes gestart: padplanning, lokalisatie, PID,"
echo "  duckie-detectie en kruispunt-navigatie."
echo "=========================================================="

dt-launchfile-join

