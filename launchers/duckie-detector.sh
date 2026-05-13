#!/bin/bash

source /environment.sh
dt-launchfile-init
rosrun my_package duckie_detector_node.py
dt-launchfile-join
