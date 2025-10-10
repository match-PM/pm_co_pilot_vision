n

## Overview

`pm_co_pilot_vision` is a ROS 2 package designed for computer vision tasks in autonomous pilot systems. It provides modules for image acquisition, processing, and object detection.

## Features

- Real-time image streaming and processing
- Object detection and classification
- Integration with ROS 2 topics and services
- Configurable parameters for vision algorithms

## Installation

Clone the repository into your ROS 2 workspace and build:

```bash
cd ~/'ros2 workspace'/src
git clone <repository-url>
cd ~/'ros2 workspace'
colcon build
```

## Usage

Launch the vision node:

```bash
ros2 launch pm_co_pilot_vision vision.launch.py
```

