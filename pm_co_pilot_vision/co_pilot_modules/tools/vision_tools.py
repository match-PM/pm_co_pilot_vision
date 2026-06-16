from langchain_core.tools import Tool
from rclpy.node import Node
from pydantic import BaseModel, Field 
from typing import List, Dict, Any
import uuid
import base64, mimetypes


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
    def __init__(self, node: Node, vision_handler: VisionHandler):

        self.node = node
        self._vision_handler = vision_handler

        # Define the actual tool using a bound method
        self.get_function_parameters_tool = Tool(
            name="get_function_parameters",
            func=self._get_function_parameters,
            description="Get the parameters and description for specific vision functions. Takes 'function_names' as input.",
            args_schema=GetFunctionParametersArgs # Specify the schema for parameters
        )

        
    def _get_function_parameters(self, function_names: List[str]) -> str:
        """
        Get the description and parameters for the requested vision functions.

        :param function_names: A list of vision function names.
        :return: A JSON string containing the parameters for the functions.
        """

        params = {}
        for function_name in function_names:
            function_params = self._vision_handler.return_function_parameters(function_name)
            if function_params:
                params[function_name] = function_params
        
        self.node.get_logger().info(f"Returning parameters for functions: {function_names}")

        return json.dumps(params, indent=2)
