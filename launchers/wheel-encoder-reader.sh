#!/bin/bash

source /environment.sh

dt-launchfile-init
rosrun my_package encoder.py
dt-launchfile-join