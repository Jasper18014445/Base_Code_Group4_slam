cmake_minimum_required(VERSION 2.8.3)
project(my_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()

catkin_install_python(PROGRAMS
  src/odometry.py
  src/slam.py
  src/sensor_fusion.py
  src/duckie_detector_node.py
  src/lane_following_duckie.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)