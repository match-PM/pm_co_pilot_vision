from langchain.tools import Tool
from rclpy.node import Node
from pydantic import BaseModel, Field 
from typing import List, Dict, Any
import uuid
import base64


from pm_co_pilot_vision.utils.vision_functions import VisionHandler

import json

# Define an empty Pydantic model for tools with no arguments
class NoArgs(BaseModel):
    """No arguments expected."""
    pass

# Define a Pydantic model for get_function_parameters (if you want to be explicit)
class GetFunctionParametersArgs(BaseModel):
    function_names: List[str] = Field(description="List of the vision functions to get description and parameters for.")

class Tools:
    """
    The Tools class provides a set of tools that can be used by the agent.
    Each tool is a function that performs a specific action.
    """
    def __init__(self, node: Node):

        self.node = node
        self._vision_handler = VisionHandler(node)

        # Define the actual tool using a bound method
        self.get_function_parameters_tool = Tool(
            name="get_function_parameters",
            func=self._get_function_parameters,
            description="Get the parameters and description for specific vision functions. Takes 'function_names' as input.",
            args_schema=GetFunctionParametersArgs # Specify the schema for parameters
        )

        self.get_available_functions_tool = Tool(
            name="get_available_functions_names",
            func=self._get_available_functions_names,
            description="Get the names of all available vision functions. This tool takes no arguments.", # Clarify in description
            args_schema=NoArgs # Explicitly state no arguments
        )

        self.load_image = Tool(
            name="LoadImage",
            func=self._load_image,
            description="Loads a base64-encoded image string into memory and returns a unique image_id. Args: {file_bytes: str} base64-encoded. Returns: {image_id: str}."
        )

        
    def _load_image(self, file_bytes: str) -> dict:
        """
        file_bytes: base64‐encoded string of image data
        returns {"image_id": "<uuid>"}
        """
        # Decode from base64
        try:
            img_data = base64.b64decode(file_bytes)
        except Exception as e:
            raise ValueError("Failed to base64-decode file_bytes") from e

        image_id = str(uuid.uuid4())
        image_path = f"/home/match-mover/Pictures/{image_id}.png"
        with open(image_path, "wb") as f:
            f.write(img_data)

        self.node.get_logger().info(f"Saved image {image_id}.png ({len(img_data)} bytes)")
        
        return {"image_id": image_id}

    # def _analyze_image(image_id: str,
    #                 analysis_types: List[str]) -> Dict[str, any]:
    #     """
    #     — image_id: the ID returned by LoadImage
    #     — analysis_types: e.g. ["histogram", "edge_map", "saliency"]
    #     Returns a JSON with exactly the stats the LLM needs.
    #     """
    #     results = {}
    #     if "histogram" in analysis_types:
    #         results["histogram_peaks"] = compute_histogram_peaks(image_id)
    #     if "edge_map" in analysis_types:
    #         results["edge_density"] = compute_edge_density(image_id)
    #     return results


    def _get_available_functions_names(self) -> str:
        """
        Get the names of all available vision functions.
        
        :return: A JSON string containing the names of all available vision functions.
        """
        self.node.get_logger().info("Getting available function names.")
        
        function_names = self._vision_handler.return_available_function_names()
        
        if not function_names:
            # self.node.get_logger().error("No available function names found.")
            return json.dumps({"error": "No available function names found."})
        
        return function_names
    

    def _get_function_parameters(self, function_names: List[str]) -> str:
        """
        Get the description and parameters for the requested vision functions.

        :param function_names: A list of vision function names.
        :return: A JSON string containing the parameters for the functions.
        """
        self.node.get_logger().info(f"Getting parameters for functions: {function_names}")

        params = {}
        for function_name in function_names:
            function_params = self._vision_handler.return_function_parameters(function_name)
            if function_params:
                params[function_name] = function_params

        return json.dumps(params, indent=2)
