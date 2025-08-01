
import rclpy
from rclpy.node import Node
import time # For potential delays or checks
import sys
import os
import cv2
import json
import uuid
from ament_index_python.packages import get_package_share_directory

from pm_vision_manager.va_py_modules.vision_pipline_processing import process_image, set_camera_parameters
from pm_vision_manager.va_py_modules.vision_assistant_class import VisionProcessClass

class VisionHandler():
    def __init__(self, vision_node: Node):
        self.vision_node = vision_node

        path = get_package_share_directory("pm_co_pilot_vision")
        vision_functions_json = f"{path}/vision_functions.json"

        with open(vision_functions_json, 'r') as f:
            self.vision_functions = json.load(f)

        self.function_names = [func['function_name'] for func in self.vision_functions]

        # Define dummy filenames for testing
        self.process_filename = "test_pipeline.json" # Replace with actual path
        self.camera_config_filename = "pm_robot_basler_top_cam_1.yaml" # Replace with actual path
        # self.image_path = "/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_image_db/test_pipeline/"
        self.image_path = "/home/match-mover/Pictures/"
        self.image_name = "test.png"

    def return_available_function_names(self) -> str:
        """
        Returns a list of available vision function names.
        """
        return json.dumps(self.function_names, indent=2, separators=(",", ":"))

    def return_function_parameters(self, function_name: str):
        """
        Returns the parameters for a given vision function as a string.
        
        :param function_name: The name of the vision function.
        :return: A dictionary of parameters for the function.
        """
        function = self.vision_functions
        for func in function:
            if func['function_name'] == function_name:
                result = {
                    "description": func.get('description', ''),
                    "params": func.get('params', [])
                }
                return json.dumps(result)
            
        return json.dumps({})


    def process_image(self, image_id: str, process_filename: str) -> tuple:
        """
        Process the given image using the defined pipeline.
        
        :param image: The input image to be processed.
        :return: The processed image and any results from the processing.
        """

        image_name = "test.png"
        # load image as image:np.ndarray
        image = cv2.imread(f"{self.image_path}/{image_name}")

        vision_processor = VisionProcessClass(
            vision_node=self.vision_node,
            process_filename=process_filename,
            camera_config_filename=self.camera_config_filename,
            launch_as_assistant=False, # Set as per your testing needs
            process_UID='CoPilotVision',
            run_cross_validation=False # Set as per your testing needs
        )

        vision_processor._load_process_file()

        set_parameter_success = set_camera_parameters(vision_processor.vision_node, 
                                                        vision_processor.image_processing_handler,
                                                        vision_processor.process_pipeline_list)

        if not set_parameter_success:
            vision_processor.image_processing_handler.vision_routine_done = True
            vision_processor.image_processing_handler.visionOK = False
            error_msg = "Failed to set camera parameters."
            self.vision_node.get_logger().error(error_msg)
            return None, error_msg


        vision_processor.image_processing_handler.set_image_metatdata(vision_processor.process_db_path,image_name)
        vision_processor.image_processing_handler.set_initial_image(image)

        #display_image = process_image(self, image, self.process_pipeline_list)
        display_image, vision_results = process_image(vision_processor.vision_node, vision_processor.image_processing_handler, vision_processor.process_pipeline_list)
        final_image = vision_processor.image_processing_handler.get_final_image()

        result_dict = vision_processor.save_vision_results()

        self.vision_node.get_logger().info(f'vision result: {result_dict}')

        # save final image
        image_name_processed = image_name.replace('.png', '_processed.png')
        cv2.imwrite(f"{self.image_path}/final_{image_name_processed}", final_image)

        return result_dict, "Processing completed successfully."