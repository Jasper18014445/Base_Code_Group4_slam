#!/bin/bash
source /environment.sh
dt-launchfile-init

echo "[slam-launch] Starten nodes..."

rosrun my_package odometry.py &
sleep 2
echo "[slam-launch] odometry gestart"

rosrun my_package slam.py &
sleep 2
echo "[slam-launch] slam gestart"

rosrun my_package sensor_fusion.py
echo "[slam-launch] sensor_fusion gestart"

dt-launchfile-join
