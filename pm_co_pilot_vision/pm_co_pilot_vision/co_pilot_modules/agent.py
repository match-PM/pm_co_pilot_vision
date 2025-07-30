import json
from rclpy.node import Node
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import uuid
import time
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import getpass
from pm_vision_app.py_modules.vision_pipeline_class import VisionPipeline
from .tools.execute_vision_pipeline import VisionPipelineTool
from .tools.vision_tools import Tools
from pm_co_pilot_vision.utils.vision_functions import VisionHandler
from ament_index_python.packages import get_package_share_directory

class Agent:
    """
    The Agent class is responsible for managing the interaction with the LLM.
    It handles the conversation history and the tools available to the LLM.
    """

    def __init__(self, node: Node):
        super().__init__()

        self.node = node
        self.node.get_logger().info("Agent __init__ called.")

        try:
            path = get_package_share_directory("pm_co_pilot_vision")
            prompt = path + '/prompts.yaml'
            with open(prompt, 'r') as file:
                config_data = yaml.safe_load(file)
                tool = config_data['agent']
                self.model = tool['model']
                self.model_provider = tool['model_provider']
                self.temperature = tool['temperature']
                self.system_prompt = tool['system_prompt']
        except Exception as e:
            self.node.get_logger().error(f"Error loading prompts.yaml: {e}")
            raise

        self.llm = ChatOpenAI(temperature=self.temperature, model=self.model)

        self.node.get_logger().info("LLM initialized.")

        self.vision_functions = VisionHandler(self.node)
        self.available_functions = self.vision_functions.return_available_function_names()

        self.vision_tools = Tools(node)

        try:
            self.tools = [
                VisionPipelineTool(node=self.node),
                self.vision_tools.get_function_parameters_tool,
                self.vision_tools.get_available_functions_tool
            ]
        except Exception as e:
            self.node.get_logger().error(f"Error initializing tools: {e}")        

        self.node.get_logger().info("Vision tools initialized.")

    def create_vision_pipeline_agent(self):
        self.node.get_logger().info("create_vision_pipeline_agent called.")
        llm = self.llm

        tools = self.tools

        # Define the prompt for the agent
        prompt = ChatPromptTemplate.from_messages(
            [
            # 1) Role & goal
            ("system", self.system_prompt),

            ("system", "Available vision functions:"),
            ("system", self.available_functions),

            # History & user input
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),

            # Scratchpad
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.node.get_logger().info(f"Prompt defined. {prompt.messages}")

        # Create the agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        self.node.get_logger().info("Agent created.")

        # Create the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        self.node.get_logger().info("AgentExecutor created.")
        return agent_executor