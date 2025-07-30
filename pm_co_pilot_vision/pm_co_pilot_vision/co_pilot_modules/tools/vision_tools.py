from langchain.tools import Tool
from rclpy.node import Node

from pm_co_pilot_vision.utils.vision_functions import VisionHandler

import json

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
            description="Get the parameters for a specific vision function."
        )

        self.get_available_functions_tool = Tool(
            name="get_available_functions_names",
            func=self.get_available_functions_names,
            description="Get the names of all available vision functions."
        )
    
    def get_available_functions_names(self) -> str:
        """
        Get the names of all available vision functions.
        
        :return: A JSON string containing the names of all available vision functions.
        """
        self.node.get_logger().info("Getting available function names.")
        
        function_names = self._vision_handler.return_available_function_names()
        
        if not function_names:
            self.node.get_logger().error("No available function names found.")
            return json.dumps({"error": "No available function names found."})
        
        return json.dumps(function_names, indent=2)
    

    def _get_function_parameters(self, function_name: str) -> str:
        """
        Get the parameters for a specific vision function.
        
        :param function_name: The name of the vision function.
        :return: A JSON string containing the parameters for the function.
        """
        self.node.get_logger().info(f"Getting parameters for function: {function_name}")
        
        params = self._vision_handler.return_function_parameters(function_name)
        
        if not params:
            self.node.get_logger().error(f"No parameters found for function: {function_name}")
            return json.dumps({"error": "Function not found or has no parameters."})
        
        return json.dumps(params, indent=2)