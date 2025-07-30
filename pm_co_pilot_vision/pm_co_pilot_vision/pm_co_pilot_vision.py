import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from PyQt5.QtWidgets import QApplication
from threading import Thread
import sys

from langchain.agents import AgentExecutor
from pm_co_pilot_vision.co_pilot_modules.agent import Agent
from pm_co_pilot_vision.utils.vision_functions import VisionHandler



class PmCoPilotVisionNode(Node):

    def __init__(self):
        super().__init__('pm_co_pilot_vision_node')
        self.agent = Agent(self)

        response = self.agent.handle_user_input("build and run a pipeline with a threshold of 128 and CircleDetection")
        self.get_logger().info(f"Agent response: {response}")




def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=6) 

    # app = QApplication(sys.argv)

    co_pilot_vision_node = PmCoPilotVisionNode()
    executor.add_node(co_pilot_vision_node)

    thread = Thread(target=executor.spin)
    thread.start()
    
    try:
        co_pilot_vision_node.get_logger().info("PmCoPilotVisionNode is running...")
        # sys.exit(app.exec())

    finally:
        co_pilot_vision_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
