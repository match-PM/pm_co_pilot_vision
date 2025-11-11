
from matplotlib import image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time # For potential delays or checks
import sys
import os
import cv2
import json
import base64, mimetypes
from datetime import datetime
from pathlib import Path
import numpy as np
import uuid
from ament_index_python.packages import get_package_share_directory
from typing import Optional, TypedDict

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data

from pm_vision_manager.va_py_modules.vision_pipline_processing import process_image, set_camera_parameters
from pm_vision_manager.va_py_modules.vision_assistant_class import VisionProcessClass

from pm_co_pilot_vision.co_pilot_modules.llm_config import LLMConfig

import re

class VisionHandler():

    RESIZE_FACTOR = 0.1  # Resize factor for testing

    def __init__(self, vision_node: Node, image_name: str, image_path: str, image_processes_path: str):
        self.vision_node = vision_node

        llm_config = LLMConfig("agent")

        self.image_name = image_name
        self.pipeline_name = image_name.replace('.png', '') + "_" + llm_config.model.replace(':', '_') + "_" + datetime.now().strftime("%Y%m%d_%H_%M_%S")
        self.image_path = image_path
        self.image_result_path = image_path + "/" + self.pipeline_name + "/"
        self.image_processes_path = image_processes_path + "/" + self.pipeline_name + "/"
        self.callback_group_RE = ReentrantCallbackGroup()

        self._debug_log_list = None

        # self.pm_robot_utils = PmRobotUtils(self)

        # Resolve vision_functions.json: prefer installed share dir, fallback to repo files/
        try:
            path = get_package_share_directory("pm_co_pilot_vision")
            vision_functions_json = f"{path}/vision_functions.json"
        except Exception as e:
            # Fallback to local repository path when running from source
            pkg_root = Path(__file__).resolve().parent.parent
            vision_functions_json = str(pkg_root / "files" / "vision_functions.json")
            try:
                self.vision_node.get_logger().warn(
                    f"Using local vision_functions.json at '{vision_functions_json}' (install not sourced?): {e}"
                )
            except Exception:
                pass

        if not os.path.exists(vision_functions_json):
            raise FileNotFoundError(f"vision_functions.json not found at {vision_functions_json}")

        with open(vision_functions_json, 'r') as f:
            self.vision_functions = json.load(f)

        self.function_names = [func['function_name'] for func in self.vision_functions]

        process_filename = self.pipeline_name + ".json"
        process_file_path = os.path.join(self.image_processes_path, process_filename)

        # Ensure directory exists
        os.makedirs(self.image_processes_path, exist_ok=True)

        process_file_content = {
                "vision_process_name": "process_name",
                "process creation": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "vision_pipeline": "pipeline_steps"
        }

        # Create empty JSON file if not exists
        if not os.path.exists(process_file_path):
            with open(process_file_path, 'w') as f:
                json.dump(process_file_content, f)

        self.process_file = f"/co_pilot_tests/{self.pipeline_name}/{self.pipeline_name}.json"


        # self.camera_config_filename = "pm_robot_basler_top_cam_1.yaml" # Replace with actual path
        # self.camera_config_filename = "pm_robot_top_cam_1_Unity.yaml" # Replace with actual path
        self.camera_config_filename = "pm_robot_basler_bottom_cam_2.yaml" 
        
        self.vision_processor = VisionProcessClass(
            vision_node=self.vision_node,
            process_filename=self.process_file,
            camera_config_filename=self.camera_config_filename,
            launch_as_assistant=False, 
            process_UID='CoPilotVision',
            run_cross_validation=False
        )


        self.vision_processor.load_camera_config()
        self.magnification = self.vision_processor.image_processing_handler.magnification
        self.pixel_size_microns = self.vision_processor.image_processing_handler.pixelsize

    def path_to_data_url(self, path: str, image_name: str, greyscale: bool, resize: bool) -> str:
        """
            Convert an image to a data URL. Use load_image with resize=True to compress the image.
            :param image_name: Name of the image file.
            :return: Data URL of the image.
            """
        image = self._load_image(path, image_name, greyscale=greyscale, resize=resize)
        if image is None:
            return ""
        
        # Encode image as PNG in memory
        success, buffer = cv2.imencode('.png', image)
        if not success:
            return ""
        mime = "image/png"
        b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


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
        try:
            function = self.vision_functions
            for func in function:
                if func['function_name'] == function_name:
                    result = {
                        "description": func.get('description', ''),
                        "quality_score": func.get('quality_score', None),
                        "params": func.get('params', [])
                    }
                    return json.dumps(result)
            self.vision_node.get_logger().error(f"Function '{function_name}' not found.")
            return json.dumps({"error": f"Function '{function_name}' not found. Use only available functions."})
        
        except Exception as e:
            self.vision_node.get_logger().error(f"Error retrieving function parameters: {e}")
            return json.dumps({"error": str(e)})

    def compute_image_facts(self, image_name: str) -> dict:
        """
        Compute various facts about the image.

        :param image_name: The name of the image file.
        :return: A JSON containing computed facts about the image.
        """
        image = self._load_image(self.image_path, image_name, greyscale=False, resize=False)
        facts = {
            "image_name": image_name,
            "dimensions": self._get_image_dimensions(image),
            # "statistics": self._get_image_statistics(image),
            "otsu_threshold": self._get_otsu_threshold(image),
            # "edge_density": self._get_edge_density(image)
            # "object_scales": self._get_object_scales(image)
        }
        self.vision_node.get_logger().info(f'dimensions: {facts["dimensions"]}')
        # self.vision_node.get_logger().info(f'statistics: {facts["statistics"]}')
        self.vision_node.get_logger().info(f'otsu_threshold: {facts["otsu_threshold"]}')
        # self.vision_node.get_logger().info(f'edge_density: {facts["edge_density"]}')
        # self.vision_node.get_logger().info(f'object_scales: {facts["object_scales"]}')

        return json.dumps(facts)

    def process_image(self, process_filename: str) -> tuple:
        """
        Process the given image using the defined pipeline.
        
        :param image: The input image to be processed.
        :return: The processed image and any results from the processing.
        """

        try:
            # load image as image:np.ndarray
            image = self._load_image(path=self.image_path, image_name=self.image_name, greyscale=False, resize=False)

            self.vision_processor.process_filename = process_filename

            self.vision_processor._load_process_file()

            set_parameter_success = set_camera_parameters(self.vision_processor.vision_node, 
                                                            self.vision_processor.image_processing_handler,
                                                            self.vision_processor.process_pipeline_list)

            if not set_parameter_success:
                self.vision_processor.image_processing_handler.vision_routine_done = True
                self.vision_processor.image_processing_handler.visionOK = False
                error_msg = "Failed to set camera parameters."
                self.vision_node.get_logger().error(error_msg)
                return None, error_msg


            self.vision_processor.image_processing_handler.set_image_metatdata(self.vision_processor.process_db_path,self.image_name)
            self.vision_processor.image_processing_handler.set_initial_image(image)

            display_image, vision_results = process_image(self.vision_processor.vision_node, self.vision_processor.image_processing_handler, self.vision_processor.process_pipeline_list)
            image_overlay = self.vision_processor.image_processing_handler.get_processed_image_overlay()
            final_image = self.vision_processor.image_processing_handler.get_final_image()
            # final_image = self.vision_processor.image_processing_handler.get_initial_image_overlay()

            # result_dict = vision_processor.save_vision_results()
            vision_result_dict = self.vision_processor.image_processing_handler.get_vision_response()

            self._debug_log_list = self.vision_processor.image_processing_handler.get_vision_process_debug()
            result_px = self._extract_and_clear_vision_logs()
            if result_px:
                self.vision_node.get_logger().info(f'Extracted vision logs: {result_px}')

            self.vision_processor.image_processing_handler.clear_vision_process_debug()
            
            quality_score = self.vision_processor.image_processing_handler.get_quality_scores()
            self.vision_processor.image_processing_handler.delete_quality_scores()

            # result_dict contains the vision_result_dict and process_debug
            result_dict = {
                "vision_result": vision_result_dict,
                "quality_score": quality_score,
                "process_debug": self._debug_log_list
            }

            self.vision_node.get_logger().warn(f'process result: {result_dict}')
            
            # save final image
            image_name_processed = self.image_name.replace('.png', '_processed.png')
            self._save_image(final_image, self.image_result_path, image_name_processed)
            # also persist overlay separately for GUI display (non-destructive)
            if image_overlay is not None:
                overlay_name = self.image_name.replace('.png', '_overlay.png')
                try:
                    self._save_image(image_overlay, self.image_result_path, overlay_name)
                except Exception as e:
                    self.vision_node.get_logger().warn(f"Failed to save overlay image: {e}")
            
            time.sleep(2)
            # save a resized version
            image_resized = self._load_image(
                path=self.image_result_path, 
                image_name=image_name_processed, 
                greyscale=False, 
                resize=True
                )
            
            self._save_image(image_resized, self.image_result_path, image_name_processed.replace('.png', '_resized.png'))

            # wait for 1 seconds to ensure image is saved
            time.sleep(1)

            # Build content blocks (processed image as data URL)
            processed_data_url = self.path_to_data_url(self.image_result_path, image_name_processed, greyscale=False, resize=True)

            self._save_result(path=self.image_result_path, result_dict=result_dict)

            content_blocks = [
                {"type": "text", "text": "Processed image and pipeline results."},
                {"type": "image_url", "image_url": {"url": processed_data_url}},
                {"type": "text", "text": f"Pipeline results: {result_dict}"}
            ]

            return content_blocks
        
        except Exception as e:
            self.vision_node.get_logger().error(f"Error processing image: {e}")
            return None, str(e)
        


    def _load_image(self, path: str, image_name: str, greyscale: bool, resize: bool) -> Optional[np.ndarray]:
        """
        Load an image with a given name.
        
        :param image_name: Name of the image file.
        :return: The loaded image as a numpy array.
        """
        image_path = Path(path, image_name).expanduser().resolve()
        if not os.path.exists(image_path):
            self.vision_node.get_logger().error(f"Image path does not exist: {image_path}")
            return None
        
        # load image as greyscale and binary
        if greyscale:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            self.vision_node.get_logger().error(f"Failed to read image at: {image_path}")
            return None

        # Compress the image for testing (resize to smaller size)
        if resize:
            image = cv2.resize(image, (0, 0), fx=self.RESIZE_FACTOR, fy=self.RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

        if image is None:
            self.vision_node.get_logger().error(f"Failed to read image at: {image_path}")
            return None
        
        return image

    def _save_image(self, image: np.ndarray, path: str, image_name: str):
        """
        Save the processed image to the specified path.
        
        :param image: The processed image as a numpy array.
        :param path: The directory path where the image will be saved.
        :param image_name: The name of the image file to save.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        image_path = Path(path, image_name).expanduser().resolve()
        cv2.imwrite(str(image_path), image)

    def _get_image_dimensions(self, image: np.ndarray) -> dict:
        height, width = image.shape[:2]
        pixel_size_um = self.pixel_size_microns
        magnification = self.magnification
        um_per_px_object = pixel_size_um / magnification

        return {
            "width": width,
            "height": height,
            "pixel_size_microns": pixel_size_um,
            "magnification": magnification,
            "um_per_px_object": um_per_px_object,               
            "fov_um": {
                "width_um": width * um_per_px_object,
                "height_um": height * um_per_px_object
            },
            "bit_depth": image.dtype.name,
            "resize_factor_preview": self.RESIZE_FACTOR         
        }

    # grayscale percentiles (p1, p50, p99), mean/std
    def _get_image_statistics(self, image: np.ndarray) -> dict:
        if len(image.shape) == 3:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        p1 = np.percentile(image, 1)
        p50 = np.percentile(image, 50)
        p99 = np.percentile(image, 99)
        mean = np.mean(image)
        std = np.std(image)

        return {
            "p1": p1,
            "p50": p50,
            "p99": p99,
            "mean": mean,
            "std": std
        }

    # global Otsu threshold candidate
    def _get_otsu_threshold(self, image: np.ndarray) -> int:
        if len(image.shape) == 3:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        otsu_threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        return otsu_threshold
    
    # edge density (Canny %, Sobel mean)
    def _get_edge_density(self, image: np.ndarray) -> dict:
        if len(image.shape) == 3:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges_canny = cv2.Canny(image, 100, 200)
        edges_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        edges_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)

        canny_density = np.sum(edges_canny > 0) / (image.size) * 100
        sobel_mean = np.mean(edges_sobel)

        return {
            "canny_density": canny_density,
            "sobel_mean": sobel_mean
        }

    # estimated object scale(s) from connected components
    def _get_object_scales(self, image: np.ndarray) -> list:
        if len(image.shape) == 3:  # Color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scales = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            scale = (w + h) / 2  # Average of width and height
            scales.append(scale)

        return scales

    def _save_result(self, path, result_dict):
        """
        Save result_dict to <path>/vision_results.json
        - Converts complex objects to dicts
        - If vision_result is a long 'pm_vision_interfaces.msg.VisionResponse(...)' string,
        extracts key=value pairs at the top level into a flat dict
        """

        def make_serializable(obj):
            """Recursively make any object JSON-serializable."""
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, "to_dict"):
                return make_serializable(obj.to_dict())
            elif hasattr(obj, "__dict__"):
                return make_serializable(obj.__dict__)
            else:
                return str(obj)

        def simple_parse_vision_string(s):
            """Very simple parser for VisionResponse(...) style strings."""
            if not isinstance(s, str) or "VisionResponse(" not in s:
                return s
            s = s[s.find("(") + 1 : s.rfind(")")]  # strip outer (...)
            matches = re.findall(r"(\w+)=([^,()]+)", s)
            return {k: v.strip().strip("'") for k, v in matches}

        # Try parsing the vision_result string if needed
        vision_result = result_dict.get("vision_result")
        if isinstance(vision_result, str):
            parsed = simple_parse_vision_string(vision_result)
            result_dict["vision_result"] = parsed

        serializable = {k: make_serializable(v) for k, v in result_dict.items()}

        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, "vision_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)

    def _extract_and_clear_vision_logs(self):
        """
        Extract vision features (circles, corners) from lines like:
        'Circle: (1295.0373, 1037.9298), radius 220.4543'
        'Corner: (100.0, 200.0)'
        and remove those lines from the handler's process debug log.
        
        Returns a dictionary:
        {
            "circles": [(x, y, r), ...],
            "corners": [(x, y), ...]
        }
        """

        # float (incl. .123, 123., +/- and scientific notation)
        FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

        pattern_circle = re.compile(
            rf"^\s*Circle:\s*\(\s*({FLOAT})\s*,\s*({FLOAT})\s*\)\s*,\s*radius\s*({FLOAT})\s*$",
            flags=re.IGNORECASE,
        )
        
        # New pattern for corners
        pattern_corner = re.compile(
            rf"^\s*Corner:\s*\(\s*({FLOAT})\s*,\s*({FLOAT})\s*\)\s*$",
            flags=re.IGNORECASE,
        )

        # get current logs from the handler (source of truth)
        logs = list(self.vision_processor.image_processing_handler.get_vision_process_debug() or [])

        # Initialize lists for both types of features
        extracted_circles: list[tuple[float, float, float]] = []
        extracted_corners: list[tuple[float, float]] = []
        kept: list[str] = []

        for log in logs:
            log_stripped = log.strip()
            
            m_circle = pattern_circle.match(log_stripped)
            if m_circle:
                x_roi, y_roi, r = map(float, m_circle.groups())
                extracted_circles.append((x_roi, y_roi, r))
                continue # Go to the next log line

            m_corner = pattern_corner.match(log_stripped)
            if m_corner:
                x, y = map(float, m_corner.groups())
                extracted_corners.append((x, y))
                continue # Go to the next log line
                
            # If neither pattern matched, keep the log
            kept.append(log)

        # overwrite the handler's log with only the kept lines
        self.vision_processor.image_processing_handler.clear_vision_process_debug()
        for log in kept:
            self.vision_processor.image_processing_handler.append_vision_process_debug(log)

        # also update your local cache
        self._debug_log_list = kept

        # Return a dictionary containing both lists
        return {"circles": extracted_circles, "corners": extracted_corners}
    

    # def _extract_and_clear_circle_logs(self):
    #     """
    #     Extract (x_roi, y_roi, r) triples from lines like:
    #     'Circle: (1295.0373, 1037.9298), radius 220.4543'
    #     and remove those lines from the handler's process debug log.
    #     Returns a list of (x, y, r) as floats.
    #     """

    #     # float (incl. .123, 123., +/- and scientific notation)
    #     FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    #     pattern = re.compile(
    #         rf"^\s*Circle:\s*\(\s*({FLOAT})\s*,\s*({FLOAT})\s*\)\s*,\s*radius\s*({FLOAT})\s*$",
    #         flags=re.IGNORECASE,
    #     )

    #     # get current logs from the handler (source of truth)
    #     logs = list(self.vision_processor.image_processing_handler.get_vision_process_debug() or [])

    #     extracted: list[tuple[float, float, float]] = []
    #     kept: list[str] = []

    #     for log in logs:
    #         m = pattern.match(log.strip())
    #         if m:
    #             x_roi, y_roi, r = map(float, m.groups())
    #             extracted.append((x_roi, y_roi, r))
    #         else:
    #             kept.append(log)

    #     # overwrite the handler's log with only the kept lines
    #     # (no need to nuke everything with clear_vision_process_debug)
    #     self.vision_processor.image_processing_handler.clear_vision_process_debug()
    #     for log in kept:
    #         # this exists on your handler per your snippet
    #         self.vision_processor.image_processing_handler.append_vision_process_debug(log)

    #     # also update your local cache if you keep using it later
    #     self._debug_log_list = kept

    #     return extracted
