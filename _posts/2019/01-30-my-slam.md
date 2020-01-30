---
layout: post
title: cartographer 运行自己的数据集
categories: [激光SLAM]
description: cartographer run demo
keywords: cartographer
---
# cartographer之rplidar、EAIydlidar、杉川雷达配置过程

## 一、配置环境（安装驱动）

1、系统环境：ubuntu16.04+ROS-kinetic

2、安装ROS-kinetic 详细说明过程 http://wiki.ros.org/kinetic/Installation/Ubuntu

## 二、代码程序

见附件

## 三、说明文档（配置步骤+代码说明）

### 3.1 配置过程

1、安装protobuf3

```
# 首先安装protobuf
sudo apt-get install autoconf autogen
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure
make
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```

检查protobuf版本

```
protoc --version
libprotoc 3.6.1
```

添加环境变量来确保连接到正确的prtobuf版本

```
which protoc
/usr/local/bin/protoc
export PROTOBUF_PROTOC_EXECUTABLE=/usr/local/bin/protoc
```

2、安装cartographer

```
# Install wstool and rosdep.
sudo apt-get update
sudo apt-get install -y python-wstool python-rosdep ninja-build

# Create a new workspace in 'catkin_ws'.
mkdir catkin_ws
cd catkin_ws
wstool init src

wstool merge -t src https://raw.githubusercontent.com/googlecartographer/cartographer_ros/master/cartographer_ros.rosinstall
wstool update -t src


# Install deb dependencies.
# The command 'sudo rosdep init' will print an error if you have already
# executed it since installing ROS. This error can be ignored.
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y

# Build and install.
catkin_make_isolated --install --use-ninja
source install_isolated/setup.bash

```

安装完成！！！

## 3.2 雷达测试

3.2.1 rplidar配置过程

```
#连接好雷达
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
```

如果没有配置过rplidar_ros下的过程，可参考hector-slam详细配置过程;

如果配置过，直接进入启动过程

3.2.2 ydlidar配置过程

如果没有配置过ydlidar_ros下的过程，可参考hector-slam之EAI雷达配置过程;

如果配置过，直接进入启动过程

3.2.3 杉川雷达配置过程

```
#连接好雷达
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0
```

如果没有配置过Delta_2A_ros下的过程，可参考hector-slam之杉川雷达配置过程;

如果配置过，直接进入启动过程

## 3.3 启动过程

修改cartographer_ros--cartographer_ros--launch--demo_revo_lds.launch

```xml
<launch>
 
  <param name="/use_sim_time" value="false" />
  <node name="cartographer_node" pkg="cartographer_ros"
      type="cartographer_node" args="
          -configuration_directory $(find cartographer_ros)/configuration_files
          -configuration_basename revo_lds.lua"
      output="screen">
    <remap from="scan" to="scan" />
  </node>
  <node name="cartographer_occupancy_grid_node" pkg="cartographer_ros"
      type="cartographer_occupancy_grid_node" args="-resolution 0.05" />
 
  <node name="rviz" pkg="rviz" type="rviz" required="true"
      args="-d $(find cartographer_ros)/configuration_files/demo_2d.rviz" />
  
</launch>
```

修改cartographer_ros--cartographer_ros--configuration_files--revo_lds.lua

```lua
include "map_builder.lua"
include "trajectory_builder.lua"
 
options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "laser",
  published_frame = "laser",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}
 
MAP_BUILDER.use_trajectory_builder_2d = true
 
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 35
TRAJECTORY_BUILDER_2D.min_range = 0.3
TRAJECTORY_BUILDER_2D.max_range = 8.
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 1.
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1
 
POSE_GRAPH.optimization_problem.huber_scale = 1e2
POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.constraint_builder.min_score = 0.65
 
return options
```

重新编译工作区间

```
catkin_make_isolated --install --use-ninja
```

3.3.1 rplidar 启动过程

```
roslaunch rplidar_ros rplidar.launch
roslaunch cartographer_ros demo_revo_lds.launch
```

3.3.2 ydlidar 启动过程

```
roslaunch ydlidar ydlidar_test.launch
roslaunch cartographer_ros demo_revo_lds.launch
```

3.3.3 杉川雷达启动过程

```
roslaunch Delta_2A_ros delta_lidar.launch
roslaunch cartographer_ros demo_revo_lds.launch
```


