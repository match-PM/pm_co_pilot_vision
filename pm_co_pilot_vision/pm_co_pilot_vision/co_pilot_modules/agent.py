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
                raw_system_prompt = tool['system_prompt']
        except Exception as e:
            self.node.get_logger().error(f"Error loading prompts.yaml: {e}")
            raise

        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}    

        self.vision_functions = VisionHandler(self.node)
        self.available_functions = self.vision_functions.return_available_function_names()

        # self.node.get_logger().info(f"raw_system_prompt: {raw_system_prompt}")

        # Inject available functions into system prompt
        self.system_prompt = raw_system_prompt.format(
            available_functions=self.available_functions
        )

        # self.node.get_logger().info(f"System prompt: {self.system_prompt}")

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            MessagesPlaceholder(variable_name="messages") # Keep this to include conversation history
        ])        

        self.vision_tools = Tools(node)

        try:
            self.tools = [
                VisionPipelineTool(node),
                self.vision_tools.get_function_parameters_tool,
                self.vision_tools.load_image
            ]
        except Exception as e:
            self.node.get_logger().error(f"Error initializing tools: {e}")        


    def handle_user_input(self, user_message: str, image_name: str = None) -> str:
        """
        Non-streaming approach (returns final string).
        """
        messages = [HumanMessage(content=user_message)]

        self.node.get_logger().info(f"Handling user input: {user_message}, image_name: {image_name}")
        if image_name is not None:
            # 3. Inject a function‐response message so the model “knows” the ID
            messages.append(FunctionMessage(
                name="LoadImage",
                content=json.dumps({"image_name": image_name})
            ))

        agent_executor = self.create_executor()
        response = agent_executor.invoke({"messages": messages}, self.config)
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
    

    # def create_vision_pipeline_agent(self):
    #     self.node.get_logger().info("create_vision_pipeline_agent called.")

    #     llm = self.llm
    #     tools = self.tools

    #     # prompt = ChatPromptTemplate.from_messages([
    #     #     ("system", "You are a helpful assistant that can create and run vision pipelines using available vision functions."),
    #     #     # ("system", "Available vision functions: {available_functions}"),
    #     #     MessagesPlaceholder(variable_name="chat_history"),
    #     #     ("user", "{input}"),
    #     #     MessagesPlaceholder(variable_name="agent_scratchpad"),
    #     # ])

    #     prompt = self.system_prompt

    #     self.node.get_logger().info("Prompt defined with available functions injected.")

    #     agent = create_openai_tools_agent(llm, tools, prompt)
    #     self.node.get_logger().info("Agent created.")

    #     # No default_values assignment here!
    #     agent_executor = AgentExecutor(
    #         agent=agent,
    #         tools=tools,
    #         verbose=True
    #     )

    #     return agent_executor

