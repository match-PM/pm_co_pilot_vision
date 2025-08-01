from matplotlib import image
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from PyQt5.QtWidgets import QApplication
from threading import Thread
import sys
import os
from langchain.agents import AgentExecutor
from pm_co_pilot_vision.co_pilot_modules.agent import Agent
from pm_co_pilot_vision.utils.vision_functions import VisionHandler



class PmCoPilotVisionNode(Node):

    def __init__(self):
        super().__init__('pm_co_pilot_vision_node')
        self.agent = Agent(self)

        vision_handler = VisionHandler(self)

        image_name = "test.png"
        # image_path = os.path.join("/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_processes/", image_name)
        image_path = os.path.join("/home/match-mover/Pictures/", image_name)

        response = self.agent.handle_user_input("Convert the image to grayscale and add a otsu threshold", image_name=image_name)
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
