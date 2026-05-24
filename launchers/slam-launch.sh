#!/bin/bash
# slam-launch.sh  –  Start alle SLAM-nodes
source /environment.sh
dt-launchfile-init

echo "[slam-launch] Starten nodes..."

# 1. Odometry
rosrun my_package odometry.py &
sleep 2
echo "[slam-launch] ✓ odometry gestart"

# 2. SLAM (bouwt kaart + publiceert /slam_map en /visual_pose)
rosrun my_package slam.py &
sleep 2
echo "[slam-launch] ✓ slam gestart"

# 3. Sensor fusion (EKF: odometry + visual_pose → /fused_pose)
rosrun my_package sensor_fusion.py &
sleep 2
echo "[slam-launch] ✓ sensor_fusion gestart"

# 4. Visualisatie: schrijft kaart elke 0.5s naar /data/slam_map.png
rosrun my_package slam_viz.py &
sleep 1
echo "[slam-launch] ✓ slam_viz gestart"
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  🗺  Kaart wordt opgeslagen naar /data/slam_map.png"
echo "  Bekijk live via: open slam_viewer.html op je laptop"
echo "  en vul het robot-IP in: $(hostname -I | awk '{print $1}')"
echo "══════════════════════════════════════════════════════════"

dt-launchfile-join
