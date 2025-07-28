from langchain.tools import Tool
from rclpy.node import Node

import json

class Tools:
    """
    The Tools class provides a set of tools that can be used by the agent.
    Each tool is a function that performs a specific action.
    """
    def __init__(self):

        # Define the actual tool using a bound method
        self.search_tool = Tool(
            name="search",
            func=self._search,
            description="Search for weather forecast"
        )