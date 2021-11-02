# Instructions

## Needed software

1. Compile (but not install) the lastest version of CMake as [here](https://github.com/IntelRealSense/librealsense/issues/6980#issuecomment-666858977) (do not follow the lastest two steps)
2. Compile the lastest version of librealsense (using the binary of the previuos compiled version of CMake) as [here](https://github.com/IntelRealSense/librealsense/issues/6964#issuecomment-787820436)
3. In the ```grasp_detector``` package folder, pull [this](https://github.com/skumra/robotic-grasping) repository.
4. Pull in the ```src``` folder of the catkin workspace [this](https://github.com/IntelRealSense/realsense-ros) repository.
5. Pull in the ```src``` folder of the catkin workspace the ''noetic'' branch of [this](https://github.com/ros-perception/vision_opencv.git) repository, changing:
```
    find_package(Boost REQUIRED python37)
```
to
```
    find_package(Boost REQUIRED python3)
```
6. Pull in the ```src``` folder of the workspace the ''gazebo-simulator-noetic'' branch of [this](https://github.com/icclab/niryo_one_ros) repository.

## Compilation

Now we have to proceed to configure the workspace for compilation.

```
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_LIBRARY=/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6.so -DPYTHON_VERSION=3
```

### Usage

1. Modify in the ```grasp_detector``` package the launch file according to: your Intel RealSense RGB-D camera ID and the path to the Network to load.
2. Modify in the ```aruco_detector``` package the config file ```aruco_parameters.yaml``` according to your robot's configuration.
3. Launch the nodes with:
```
    roslaunch niryo_talker niryo_grasping.launch
```

**Note well**, this will launch all the nodes needed to provide the ```compute_grasping_pose``` service. To perform grasps, you have to run your client on the Niryo One.