import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # Define a ROS 2 node
    pm_co_pilot_vision_app_node = Node(
        package='pm_co_pilot_vision',
        executable='pm_co_pilot_vision',
        name='PM_Co_Pilot_Vision',
        #parameters=[{'param_name': LaunchConfiguration('my_param')}]
        emulate_tty=True
    )
    
    ld = LaunchDescription()
    ld.add_action(pm_co_pilot_vision_app_node)

    return ld