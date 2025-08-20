import json
import os
from rclpy.node import Node
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, field_validator
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import uuid
import time
from pathlib import Path
import getpass

from pm_vision_app.py_modules.vision_pipeline_class import VisionPipeline
from pm_vision_app.py_modules.vision_functions_loader import VisionFunctionsLoader
from pm_co_pilot_vision.utils.vision_functions import VisionHandler


class VisionPipelineInput(BaseModel):
    """Input for creating and running a vision pipeline."""
    pipeline_steps: List[Dict[str, Any]] = Field(
        ..., description="A json of vision function dictionaries, each containing the function name as key and its parameters as values. Ensure 'active' is set for each step. Example: [{'BGR2GRAY': {'active': True}}, {'Threshold': {'active': True, 'thresh': 128, 'maxval': 255, 'type': 'THRESH_BINARY'}}]")
    process_name: str = Field(
        "default_pipeline", description="A descriptive name for the vision pipeline process file."
    )

class VisionPipelineTool(BaseTool):
    # allow arbitrary types to stash the VisionPipeline
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # **Annotated** fields to override the BaseTool defaults
    name: str = "create_and_run_vision_pipeline"
    description: str = "Creates a vision pipeline JSON and runs it."
    args_schema: type = VisionPipelineInput

    # A private attr to hold your VisionPipeline instance
    _node: Node = PrivateAttr()
    _vision_handler: VisionHandler = PrivateAttr()

    def __init__(self, node: Node, vision_handler: VisionHandler):
        super().__init__()
        self._node = node
        self._vision_handler = vision_handler

        self._node.get_logger().info(f"VisionPipelineTool initialized with node: {self._node.get_name()}")


    def _run(self, pipeline_steps: List[Dict[str, Any]], process_name: str) -> str:
        self._node.get_logger().info(f"Creating and running vision pipeline: {process_name}")
        
        self._node.get_logger().info(f"Pipeline steps: {pipeline_steps}")
        # Normalize the pipeline steps
        pipeline_steps = self.convert_function_list(pipeline_steps)
        self._node.get_logger().info(f"Normalized pipeline steps: {pipeline_steps}")


        process_file_content = {
                "vision_process_name": process_name,
                "process creation": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "vision_pipeline": pipeline_steps
        }

        process_file_name = f"{process_name}.json"
        process_file_path = os.path.join(self._vision_handler.image_processes_path, process_file_name)
        # process_file_path = os.path.join("/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_processes/", process_file_name)
        # process_file_path = os.path.join("/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_processes/", process_file_name)

        # os.makedirs(vision_pipeline.vision_process_libary_path, exist_ok=True)

        with open(process_file_path, 'w') as f:
            json.dump(process_file_content, f, indent=4)

        # Execute the vision pipeline
        try:
            result_dict, message = self._vision_handler.process_image(
                process_filename=process_file_name
            )
        except Exception as e:
            return f"Error processing image: {e}"

        if result_dict is None:
            return f"Vision pipeline '{process_name}' error: {message}"
        else:
            return f"Vision pipeline '{process_name}' Results: {result_dict}"

    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool does not support async")


    def convert_function_list(self,input_list):
        """
        Converts a list of dictionaries from the format:
        [{"function_name": "Name", "params": {...}}]
        to:
        [{"Name": {...}}]

        Args:
            input_list (list): The list of dictionaries to convert.

        Returns:
            list: The converted list of dictionaries.
        """
        converted_list = []
        try:
            for item in input_list:
                # Accept both "function_name" or "function"
                function_key = None
                if "function_name" in item:
                    function_key = "function_name"
                elif "function" in item:
                    function_key = "function"
                elif "name" in item:
                    function_key = "function"
                # Accept both "params" or "parameters"
                params_key = None
                if "params" in item:
                    params_key = "params"
                elif "parameters" in item:
                    params_key = "parameters"
                if function_key and params_key:
                    function_name = item[function_key]
                    params = item[params_key]
                    converted_list.append({function_name: params})
                else:
                    self._node.get_logger().error(f"Invalid item format: {item}")
                    raise ValueError(f"Invalid item format: {item}")
        except Exception as e:
            self._node.get_logger().error(f"Error converting function list: {e}")
            raise
        return converted_list
