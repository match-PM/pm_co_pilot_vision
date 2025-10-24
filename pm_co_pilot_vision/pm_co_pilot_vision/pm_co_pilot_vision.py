from matplotlib import image
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from PyQt5.QtWidgets import QApplication
from threading import Thread
import sys
import os
import cv2
from langchain.agents import AgentExecutor
from pm_co_pilot_vision.co_pilot_modules.agent import Agent
from pm_co_pilot_vision.utils.vision_functions import VisionHandler
from pm_co_pilot_vision.co_pilot_modules.tools.image_describer import ImageDescriberTool
from time import sleep


class PmCoPilotVisionNode(Node):

    def __init__(self):
        super().__init__('pm_co_pilot_vision_node')

        # image_path = os.path.join("/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_processes/", image_name)
   
        # image_path = "/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_db/test_pipeline/"
        # image_processes_path = "/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_processes"
        # image_path = "/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_image_db/co_pilot_tests/"
        # image_processes_path = "/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_processes/co_pilot_tests"
        # image_path = "/home/pmlab/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_image_db/test_pipeline/"
        # image_processes_path = "/home/pmlab/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_processes"
        image_path = "/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_db/co_pilot_tests/"
        image_processes_path = "/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_processes/co_pilot_tests"

        # image_name = "fitting_hole.png"
        image_name = "sensor_corner.png"
        # image_name = "vacuum_gripper.png"

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
        
        # user_prompt = "Convert the image to grayscale and add a threshold"
        # user_prompt = "What do you see in the image? Do not perform any operations on the image, just describe it."
        # user_prompt = "detect the circle based on the outer edge of the fitting hole"
        # user_prompt = "Detect the outer white circle of the vacuum_gripper."
        user_prompt = "Detect the corner of the sensor in the image."
        # user_prompt = "what do you see in the image? The outer edge of the hole should be detected. Was it detected correctly? Do not perform any operations on the image, just describe it."

        # image_describer_tool = ImageDescriberTool(
        #     node=self,
        #     vision_handler=vision_handler
        # )
        # response = image_describer_tool._run(user_request=user_prompt, image_name=image_name)
        # self.get_logger().info(f"ImageDescriberTool response: {response}")

        response = self.agent.handle_user_input(user_prompt, image_name=image_name)
        self.get_logger().info(f"Agent response: {response}")



def main(args=None):
    rclpy.init(args=args)

    node = PmCoPilotVisionNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)

    try:
        node.get_logger().info("PmCoPilotVisionNode is running...")
        executor.spin()  # blocks until shutdown (Ctrl-C)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


# def main(args=None):
#     rclpy.init(args=args)
#     executor = MultiThreadedExecutor(num_threads=6) 

#     # app = QApplication(sys.argv)

#     co_pilot_vision_node = PmCoPilotVisionNode()
#     executor.add_node(co_pilot_vision_node)

#     thread = Thread(target=executor.spin)
#     thread.start()
    
#     try:
#         co_pilot_vision_node.get_logger().info("PmCoPilotVisionNode is running...")
#         # sys.exit(app.exec())

#     finally:
#         co_pilot_vision_node.destroy_node()
#         executor.shutdown()
#         rclpy.shutdown()

if __name__ == '__main__':
    main()
