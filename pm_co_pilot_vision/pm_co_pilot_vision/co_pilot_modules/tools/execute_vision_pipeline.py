import json
import os
from rclpy.node import Node
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
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
        ..., description="A list of vision function dictionaries, each containing the function name as key and its parameters as values. Ensure 'active' is set for each step. Example: [{'BGR2GRAY': {'active': True}}, {'Threshold': {'active': True, 'thresh': 128, 'maxval': 255, 'type': 'THRESH_BINARY'}}]")
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

    def __init__(self, node: Node):
        super().__init__()
        self._node = node
        self._vision_handler = VisionHandler(self._node)

        self._node.get_logger().info(f"VisionPipelineTool initialized with node: {self._node.get_name()}")


    def _run(self, pipeline_steps: List[Dict[str, Any]], process_name: str) -> str:

        process_file_content = {
                "vision_process_name": process_name,
                "process creation": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "vision_pipeline": pipeline_steps
        }

        process_file_name = f"{process_name}.json"
        process_file_path = os.path.join("/home/match-mover/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_processes/", process_file_name)

        # os.makedirs(vision_pipeline.vision_process_libary_path, exist_ok=True)

        with open(process_file_path, 'w') as f:
            json.dump(process_file_content, f, indent=4)

        return True
        # Execute the vision pipeline
        # try:
        #     result_dict, message = self._vision_handler.process_image(
        #         image_name="cube.png",  # Replace with actual image name
        #         process_filename=process_file_name
        #     )
        # except Exception as e:
        #     return f"Error processing image: {e}"

        # if result_dict is None:
        #     return f"Vision pipeline '{process_name}' executed with error: {message}"
        # else:
        #     return f"Vision pipeline '{process_name}' executed successfully! Results: {result_dict}"

    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool does not support async")