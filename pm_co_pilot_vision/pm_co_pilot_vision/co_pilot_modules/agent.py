import json
from rclpy.node import Node
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
import os
import base64
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
    The Agent class is responsible for managing the interaction with the tools.
    It handles the conversation history and the tools available to the LLM.
    """

    def __init__(self, node: Node, vision_handler: VisionHandler):
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
                raw_system_prompt = tool['system_prompt']
        except Exception as e:
            self.node.get_logger().error(f"Error loading prompts.yaml: {e}")
            raise

        example_json = path + '/example_pipeline.json'

        try:
            with open(example_json, 'r') as file:
                example_pipeline = json.load(file)
        except Exception as e:
            self.node.get_logger().error(f"Error loading example_pipeline.json: {e}")
            raise

        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}    

        self.vision_functions = vision_handler
        self.available_functions = self.vision_functions.return_available_function_names()

        # Inject available functions into system prompt
        # self.system_prompt = raw_system_prompt.format(
        #     available_functions=self.available_functions
        #     # example_pipeline=json.dumps(example_pipeline, indent=2, separators=(",", ":"))
        # )


        # self.prompt = ChatPromptTemplate.from_messages(
        #     [ SystemMessagePromptTemplate.from_template(self.system_prompt), 
        #      MessagesPlaceholder(variable_name="messages") # Keep this to include conversation history 
        # ])

        # Create a SystemMessagePromptTemplate using Jinja2
        system_message_template = SystemMessagePromptTemplate.from_template(
            raw_system_prompt,
            template_format="jinja2" 
        )

        # Build the full ChatPromptTemplate with conversation history
        base_prompt = ChatPromptTemplate.from_messages([
            system_message_template,
            MessagesPlaceholder(variable_name="messages")
        ])

        funcs_str = "```json\n" + self.available_functions + "\n```"
        self.prompt = base_prompt.partial(available_functions=funcs_str)

        self.node.get_logger().info(f"System Prompt: {self.prompt}")

        self.vision_tools = Tools(node, vision_handler=vision_handler)

        try:
            self.tools = [
                VisionPipelineTool(node, vision_handler=vision_handler),
                self.vision_tools.get_function_parameters_tool
            ]
        except Exception as e:
            self.node.get_logger().error(f"Error initializing tools: {e}")        


    def handle_user_input(self, user_message: str, image_name: str = None) -> str:
        """
        Non-streaming approach (returns final string).
        """

        # Validate image
        if not image_name:
            self.node.get_logger().error("Valid image name is required for processing.")
            return "Valid image name is required for processing."

        # Build multimodal content blocks
        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]

        data_url = self.vision_functions.path_to_data_url(image_name, resize=True)
        content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

        # Compute concise image facts once
        image_facts = self.vision_functions.compute_image_facts(image_name)
        content_blocks.append({"type": "text", "text": f"Image facts: {image_facts}"})

        agent_executor = self.create_executor()

        response = agent_executor.invoke(
            {
                "messages": [HumanMessage(content=content_blocks)]
            }, 
            self.config
        )
        
        ai_message = response["messages"][-1]   # The last message should be the response
        self.node.get_logger().info(f"Agent response: {ai_message}")
        response = ai_message.content
        return response
    
    def create_executor(self):
        return create_react_agent(
            self.model,
            self.tools,
            checkpointer=self.memory,
            prompt=self.prompt
        )
    