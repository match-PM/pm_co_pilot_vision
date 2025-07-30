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

class Agent:
    """
    The Agent class is responsible for managing the interaction with the LLM.
    It handles the conversation history and the tools available to the LLM.
    """

    def __init__(self, node: Node):
        super().__init__()

        self.node = node
        self.node.get_logger().info("Agent __init__ called.")

        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.node.get_logger().info("LLM initialized.")

        vision_pipeline_config_path = "/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_functions"

        self.vision_functions = VisionPipeline(vision_pipeline_config_path)

        self.vision_handler = VisionHandler(node)

        self.vision_tools = Tools(node)
        self.node.get_logger().info("Vision tools initialized.")

        try:
            self.tools = [VisionPipelineTool(vision_functions=self.vision_functions, vision_handler=self.vision_handler), 
                        self.vision_tools.get_function_parameters_tool]
        except Exception as e:
            self.node.get_logger().error(f"Error initializing tools: {e}")

    def create_vision_pipeline_agent(self):
        self.node.get_logger().info("create_vision_pipeline_agent called.")
        llm = self.llm

        if not hasattr(self, 'tools') or not self.tools:
            self.node.get_logger().error("Error: self.tools not initialized or is empty. Agent cannot be created.")
            # You might want to raise an error or handle this case appropriately
            raise AttributeError("'Agent' object has no attribute 'tools' or tools are not properly initialized.")


        tools = self.tools

        # Define the prompt for the agent
        prompt = ChatPromptTemplate.from_messages(
            [
                # 1) Role & goal
                ("system", "You are an expert vision‑pipeline builder. Your goal is to translate user requests into a sequence of vision function calls."),

                # 2) Structure example (note all braces doubled)
                ("system", "For reference, a two‑step grayscale+threshold pipeline should look like:\n"
                        "```json\n"
                        "{{\n"
                        "  \"BGR2GRAY\": {{ \"active\": true }},\n"
                        "  \"Threshold\": {{ \"active\": true, \"thresh\": 128, \"maxval\": 255, \"type\": \"THRESH_BINARY\" }}\n"
                        "}}\n"
                        "```"),

                # 3) Directory context
                ("system", "All process JSONs will be written to `/home/match-mover/Documents/`."),
                
                # 4) JSON‑tool reminder
                ("system", "You have access to the `create_and_run_vision_pipeline` tool. **When** you call it, you **must** output **only** a JSON object with exactly two keys:\n"
                        "1. `pipeline_steps` — a **list** of step‑dicts, each mapping a function name → its params (always include `\"active\": true`).\n"
                        "2. `process_name` — a **string** to name this pipeline run.\n"
                        "No extra fields, no natural‑language—**just** the JSON."),

                # 5) Full valid‑call example (also escaped)
                ("system", "Here’s an example of a **complete valid** tool invocation:\n"
                        "```json\n"
                        "{{\n"
                        "  \"pipeline_steps\": [\n"
                        "    {{ \"CircleDetection\": {{ \"active\": true, \"draw_circles\": false, \"mode\": \"default\", \"minRadius\": 0, \"maxRadius\": 0 }} }},\n"
                        "    {{ \"Threshold\":       {{ \"active\": true, \"thresh\": 128, \"maxval\": 255, \"type\": \"THRESH_BINARY\" }} }}\n"
                        "  ],\n"
                        "  \"process_name\": \"circle_detection_threshold\"\n"
                        "}}\n"
                        "```"),

                # 6) History & user input
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),

                # 7) Scratchpad
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.node.get_logger().info("Prompt defined.")

        # Create the agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        self.node.get_logger().info("Agent created.")

        # Create the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        self.node.get_logger().info("AgentExecutor created.")
        return agent_executor