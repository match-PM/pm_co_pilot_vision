import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from PyQt6.QtWidgets import QApplication
from threading import Thread
import sys
import yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from pm_co_pilot_vision.gui.agent_gui import AgentWindow
from pm_co_pilot_vision.utils.vision_functions import VisionHandler
from pm_co_pilot_vision.co_pilot_modules.agent import Agent


class PmCoPilotVisionNode(Node):
    """Node that runs the agent with configuration from node_config.yaml."""
    def __init__(self):
        super().__init__('pm_co_pilot_vision_node')
        self.get_logger().info('Starting PmCoPilotVisionNode in node mode...')

        # Load configuration from YAML file
        config_path = Path(get_package_share_directory('pm_co_pilot_vision')) / 'config' / 'node_config.yaml'
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                node_config = config['node_mode']
                
            image_path = node_config['image_path']
            image_processes_path = node_config['image_processes_path']
            image_name = node_config['image_name']
            user_prompt = node_config['user_prompt']
            
            self.get_logger().info(f'Loaded config from: {config_path}')
            self.get_logger().info(f'Image: {image_name}')
            self.get_logger().info(f'Prompt: {user_prompt}')
            
        except FileNotFoundError:
            self.get_logger().error(f'Config file not found: {config_path}')
            raise
        except KeyError as e:
            self.get_logger().error(f'Missing configuration key: {e}')
            raise

        vision_handler = VisionHandler(
            vision_node=self, 
            image_name=image_name,
            image_path=image_path, 
            image_processes_path=image_processes_path
        )

        self.agent = Agent(
            self, 
            vision_handler=vision_handler
        )

        response = self.agent.handle_user_input(user_prompt, image_name=image_name)
        self.get_logger().info(f"Agent response: {response}")



def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)
    
    # Parse command-line arguments to determine mode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'node'],
                        help='Run mode: gui (default) or node (headless with hardcoded prompt)')
    parsed_args, _ = parser.parse_known_args()
    
    if parsed_args.mode == 'node':
        # Node mode: run the agent with hardcoded settings
        executor = MultiThreadedExecutor(num_threads=2)
        co_pilot_vision_node = PmCoPilotVisionNode()
        executor.add_node(co_pilot_vision_node)
        
        try:
            executor.spin()
        finally:
            co_pilot_vision_node.destroy_node()
            executor.shutdown()
            rclpy.shutdown()
    else:
        # GUI mode: launch the PyQt6 interface
        gui_node = rclpy.create_node('pm_co_pilot_vision_gui_node')
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(gui_node)

        # Spin ROS in a background thread
        spin_thread = Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        # Start Qt application and show GUI bound to the existing ROS node
        app = QApplication(sys.argv)
        window = AgentWindow(node=gui_node)
        window.show()

        try:
            app.exec()
        finally:
            # Shutdown ROS nicely when the GUI exits
            executor.shutdown()
            if gui_node is not None:
                gui_node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
