This branch was designed to work with ros2 humble. It will also work with ros2 foxy. Support for these environments are limited to Linux 22.X and 20.X, respectively. They rely on what is now known as Gazebo classic. The foxy version running on 20.X virtualizaed on arch architecture (Apple m chipsets) has been verified.

The ROS2 tools here depend on

ros2-humble-desktop - basic humble install
python3-colcon-common-extensions - colcon
python3-opencv - opencv
ros-humble-navigation2 - nav2 library
ros-humble-nav2-bringup - nav2 bringup tools
ros-humble-xacro - support for xacro
You will need have these installed prior to many of the packages building successfully.



This project (Project 1) has a version of the bug0 algorithm that assumes that the world is an infinite plane with circular obstacles at known locations and with know radii. The algorithm I made should leave circles at tangent points that provide a direct line to the goal location. 
