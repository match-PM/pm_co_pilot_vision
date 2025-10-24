from datetime import datetime
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
from .llm_config import LLMConfig
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

        # try:
        #     path = get_package_share_directory("pm_co_pilot_vision")
        #     prompt = path + '/prompts.yaml'
        #     with open(prompt, 'r') as file:
        #         config_data = yaml.safe_load(file)
        #         tool = config_data['agent']
        #         self.model = tool['model']
        #         self.model_provider = tool['model_provider']
        #         self.temperature = tool['temperature']
        #         raw_system_prompt = tool['system_prompt']
        # except Exception as e:
        #     self.node.get_logger().error(f"Error loading prompts.yaml: {e}")
        #     raise

        llm_config = LLMConfig("agent")
        self.model = llm_config.model
        self.model_provider = llm_config.model_provider
        raw_system_prompt = llm_config.system_prompt
        self.temperature = llm_config.temperature

        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}    

        self.vision_handler = vision_handler
        self.available_functions = self.vision_handler.return_available_function_names()

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
        # self.prompt = base_prompt.partial(resize_factor_preview=VisionHandler.RESIZE_FACTOR)

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

        data_url = self.vision_handler.path_to_data_url(
            self.vision_handler.image_path, 
            image_name, 
            greyscale=True, 
            resize=True
            )
        content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

        # Compute concise image facts once
        image_facts = self.vision_handler.compute_image_facts(image_name)
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
        self.save_token_usage_summary(response, self.vision_handler.image_processes_path)
        response = ai_message.content
        return response
    
    def create_executor(self):
        return create_react_agent(
            self.model,
            self.tools,
            checkpointer=self.memory,
            prompt=self.prompt
        )
    
    def save_token_usage_summary(self, response: Dict[str, Any], save_path: str):
        """
        Extracts token usage data from the agent response and saves it as a 
        timestamped JSON file.

        It checks multiple locations to find the usage data:
        1. ai_message.usage_metadata (standardized LangChain attribute)
        2. ai_message.response_metadata['token_usage'] (OpenAI format)
        3. ai_message.response_metadata['usage'] (Anthropic/Claude format)

        Args:
            response: The full dictionary response from agent_executor.invoke().
            save_path: The directory path where the summary file will be saved.
        """
        
        # 1. Get the last AI message from the response dictionary
        try:
            # The last message should be the AIMessage with the metadata
            ai_message = response["messages"][-1]
        except (KeyError, IndexError, TypeError):
            self.node.get_logger().warn(
                "Could not find last message in 'response[\"messages\"]'. "
                "Cannot save token summary."
            )
            return

        # 2. Extract the token usage data from various possible locations.
        usage_data = None
        
        # Try standard 'usage_metadata' attribute first
        # This is the preferred, standardized LangChain location
        if hasattr(ai_message, 'usage_metadata') and ai_message.usage_metadata:
            usage_data = ai_message.usage_metadata
            self.node.get_logger().info("Found token data in 'ai_message.usage_metadata'.")
        
        # If not found, check model-specific 'response_metadata'
        elif (hasattr(ai_message, 'response_metadata') and 
            isinstance(ai_message.response_metadata, dict)):
            
            # Check for OpenAI format
            if 'token_usage' in ai_message.response_metadata:
                usage_data = ai_message.response_metadata.get('token_usage')
                self.node.get_logger().info("Found token data in 'ai_message.response_metadata.token_usage' (OpenAI format).")
                
            # Check for Anthropic/Claude format
            elif 'usage' in ai_message.response_metadata:
                usage_data = ai_message.response_metadata.get('usage')
                self.node.get_logger().info("Found token data in 'ai_message.response_metadata.usage' (Anthropic format).")

        # 3. If still no data, log a warning and exit.
        if usage_data is None:
            self.node.get_logger().warn(
                "Could not find token usage data in 'ai_message.usage_metadata', "
                "'response_metadata.token_usage', or 'response_metadata.usage'. "
                "Cannot save token summary."
            )
            return

        # 2. Ensure the save directory exists.
        try:
            os.makedirs(save_path, exist_ok=True)
        except OSError as e:
            self.node.get_logger().error(
                f"Failed to create directory at {save_path}: {e}"
            )
            return

        # 3. Create a unique filename with a timestamp.
        filename = f"token_usage.json"
        full_save_path = os.path.join(save_path, filename)

        # 4. Write the data to the JSON file.
        try:
            with open(full_save_path, 'w') as f:
                json.dump(usage_data, f, indent=4)
                
            self.node.get_logger().info(
                f"Token usage summary saved to: {full_save_path}"
            )
        except IOError as e:
            self.node.get_logger().error(
                f"Failed to write token summary to {full_save_path}: {e}"
            )
        except Exception as e:
            self.node.get_logger().error(
                f"An unexpected error occurred while saving token summary: {e}"
            )