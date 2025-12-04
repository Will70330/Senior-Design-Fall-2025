from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
import os
import sys

def generate_launch_description():
    # Get the current directory (where this launch file is)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Path to the python script
    script_path = os.path.join(current_dir, 'ros_testing_camera.py')
    
    # Path to the RViz config
    rviz_config_path = os.path.join(current_dir, 'camera_config.rviz')

    # Setup Environment for Publisher
    # We need to ensure the venv site-packages are in PYTHONPATH so pyrealsense2 can be found
    # while running with the system python (which has rclpy).
    venv_site_packages = os.path.join(current_dir, 'venv', 'lib', 'python3.12', 'site-packages')
    
    pub_env = os.environ.copy()
    if 'PYTHONPATH' in pub_env:
        pub_env['PYTHONPATH'] = venv_site_packages + os.pathsep + pub_env['PYTHONPATH']
    else:
        pub_env['PYTHONPATH'] = venv_site_packages

    # Define the 'mode' argument
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='pub',
        description='Operation mode: "pub" for Publisher (Sender), "sub" for Subscriber (RViz Viewer)'
    )

    # Define the Publisher Node (runs the python script in 'pub' mode)
    pub_node = Node(
        executable=sys.executable,
        arguments=[script_path, 'pub'],
        output='screen',
        env=pub_env,
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'pub'"]))
    )

    # Define the RViz Node (runs standard rviz2 with our config)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('mode'), "' == 'sub'"]))
    )

    return LaunchDescription([
        mode_arg,
        pub_node,
        rviz_node,
        LogInfo(msg=["Launching in mode: ", LaunchConfiguration('mode')])
    ])
