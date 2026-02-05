import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # Declare launch argument for mode selection
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='gui',
        description='Run mode: gui (default) or node (headless with hardcoded prompt)'
    )
    
    # Define a ROS 2 node
    pm_co_pilot_vision_app_node = Node(
        package='pm_co_pilot_vision',
        executable='pm_co_pilot_vision',
        name='PM_Co_Pilot_Vision',
        arguments=['--mode', LaunchConfiguration('mode')],
        emulate_tty=True,
        output='screen'
    )
    
    ld = LaunchDescription()
    ld.add_action(mode_arg)
    ld.add_action(pm_co_pilot_vision_app_node)

    return ld