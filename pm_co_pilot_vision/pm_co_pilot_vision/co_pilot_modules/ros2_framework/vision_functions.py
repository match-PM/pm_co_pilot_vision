
import rclpy
from rclpy.node import Node
import time # For potential delays or checks
import sys
import os
import cv2

from pm_vision_manager.va_py_modules.vision_pipline_processing import process_image, set_camera_parameters
from pm_vision_manager.va_py_modules.vision_assistant_class import VisionProcessClass

class VisionFunctions():
    def __init__(self, vision_node: Node):
        self.vision_node = vision_node

        # Define dummy filenames for testing
        self.process_filename = "test_pipeline.json" # Replace with actual path
        self.camera_config_filename = "pm_robot_top_cam_1_Unity.yaml" # Replace with actual path
        self.image_path = "/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_image_db/test_pipeline/"
        self.image_name = "target.png"

    def process_image(self, image_name: str, process_filename: str) -> tuple:
        """
        Process the given image using the defined pipeline.
        
        :param image: The input image to be processed.
        :return: The processed image and any results from the processing.
        """

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