import json
import os
from rclpy.node import Node
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr, root_validator, validator
from langchain_core.tools import BaseTool
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

    @root_validator(pre=True)
    def accept_pipeline_aliases(cls, values):
        if isinstance(values, dict) and "pipeline_steps" not in values:
            for alias in ("vision_pipeline", "pipeline", "steps"):
                if alias in values:
                    values["pipeline_steps"] = values[alias]
                    break
        return values

    @validator("pipeline_steps", pre=True)
    def parse_pipeline_steps(cls, value):
        if isinstance(value, str):
            return json.loads(value)
        return value


class VisionPipelineTool(BaseTool):
    # **Annotated** fields to override the BaseTool defaults
    name: str = "create_and_run_vision_pipeline"
    description: str = "Creates a vision pipeline JSON and runs it."
    args_schema: type = VisionPipelineInput

    # A private attr to hold your VisionPipeline instance
    _node: Node = PrivateAttr()
    _vision_handler: VisionHandler = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, node: Node, vision_handler: VisionHandler):
        super().__init__()
        self._node = node
        self._vision_handler = vision_handler

        self._node.get_logger().info(f"VisionPipelineTool initialized with node: {self._node.get_name()}")


    def _run(
        self,
        pipeline_steps: Optional[List[Dict[str, Any]]] = None,
        process_name: str = "default_pipeline",
        **kwargs,
    ) -> str:
        if pipeline_steps is None:
            pipeline_steps = self._pipeline_steps_from_kwargs(kwargs)

        self._node.get_logger().info(f"Creating and running vision pipeline: {process_name}")
        
        self._node.get_logger().info(f"Pipeline steps: {pipeline_steps}")

        try:
            pipeline_steps = self._normalize_pipeline_steps(pipeline_steps)
        except Exception as e:
            self._node.get_logger().error(f"Invalid pipeline steps: {e}")
            return self._tool_response(
                ok=False,
                process_name=process_name,
                message=f"Invalid pipeline steps: {e}",
            )

        self._node.get_logger().info(f"Pipeline steps after conversion: {pipeline_steps}")

        process_file_content = {
                "vision_process_name": process_name,
                "process creation": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "vision_pipeline": pipeline_steps
        }

        # # process_file_name = f"{process_name}.json"
        # process_file_name = "test_pipeline.json"
        # process_file_path = os.path.join(self._vision_handler.image_processes_path, process_file_name)
        # # process_file_path = os.path.join("/home/match-pm/pm_Server/01_PM_Zelle/03_PM_DataBase/pm_assembly_database/Vision/vision_processes/", process_file_name)
        # # process_file_path = os.path.join("/home/match-pm/ros2_ws_pm/src/pm_vision_manager/pm_vision_manager/vision_processes/", process_file_name)

        # # os.makedirs(vision_pipeline.vision_process_libary_path, exist_ok=True)

        # with open(process_file_path, 'w') as f:
        #     json.dump(process_file_content, f, indent=4)

        process_file_name = self._vision_handler.pipeline_name + ".json"
        process_file_path = os.path.join(self._vision_handler.image_processes_path, process_file_name)

        if not os.path.exists(self._vision_handler.image_processes_path):
            os.makedirs(self._vision_handler.image_processes_path)

        with open(process_file_path, 'w') as f:
            json.dump(process_file_content, f, indent=4)

        # Execute the vision pipeline
        try:
            result_dict, message, final_image = self._vision_handler.process_image(
                process_filename=process_file_name
            )
        except Exception as e:
            self._node.get_logger().error(f"Error processing image: {e}")
            return self._tool_response(
                ok=False,
                process_name=process_name,
                message=f"Error processing image: {e}",
            )

        if result_dict is None:
            return self._tool_response(
                ok=False,
                process_name=process_name,
                message=message,
                processed_image=final_image,
            )

        return self._tool_response(
            ok=True,
            process_name=process_name,
            message=message,
            result=result_dict,
            processed_image=final_image,
        )


    def _pipeline_steps_from_kwargs(self, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("pipeline_steps", "vision_pipeline", "pipeline", "steps"):
            if key in kwargs:
                return kwargs[key]
        raise ValueError("Missing required pipeline_steps argument.")

    def _normalize_pipeline_steps(self, pipeline_steps: Any) -> List[Dict[str, Any]]:
        if isinstance(pipeline_steps, str):
            pipeline_steps = json.loads(pipeline_steps)

        if not isinstance(pipeline_steps, list):
            raise ValueError("pipeline_steps must be a list of function-step dictionaries.")

        if not self.check_pipeline_steps(pipeline_steps):
            self._node.get_logger().warn("Converting pipeline steps from function-name format.")
            pipeline_steps = self.convert_function_list(pipeline_steps)

        return [self._normalize_step(step) for step in pipeline_steps]

    def _normalize_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(f"Each pipeline step must contain exactly one function: {step}")

        function_name, params = next(iter(step.items()))
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"Parameters for '{function_name}' must be a dictionary.")

        normalized_params = dict(params)
        normalized_params = self._normalize_common_param_aliases(function_name, normalized_params)
        if function_name == "CircleDetection":
            normalized_params = self._normalize_circle_detection_params(normalized_params)

        return {function_name: normalized_params}

    def _normalize_common_param_aliases(
        self,
        function_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        if function_name == "Threshold":
            self._move_param(params, "threshold", "thresh")
            self._move_param(params, "threshold_value", "thresh")
            self._move_param(params, "max_value", "maxval")
            self._move_param(params, "maxValue", "maxval")
            self._move_param(params, "threshold_type", "type")
            self._move_param(params, "thresholdType", "type")

        if function_name in {
            "Morphology_Ex_Closing",
            "Morphology_Ex_Opening",
            "Morphology_Ex_Gradient",
            "Dilation",
            "Errosion",
        }:
            self._move_param(params, "kernel_size", "kernelsize")
            self._move_param(params, "kernelSize", "kernelsize")

        if function_name in {
            "Morphology_Ex_Closing",
            "Morphology_Ex_Opening",
            "Morphology_Ex_Gradient",
        }:
            params.pop("iterations", None)

        if function_name == "ROI":
            self._move_param(params, "center_x", "ROI_center_x")
            self._move_param(params, "center_y", "ROI_center_y")
            self._move_param(params, "height", "ROI_height")
            self._move_param(params, "width", "ROI_width")

        return params

    def _move_param(self, params: Dict[str, Any], old_key: str, new_key: str) -> None:
        if old_key in params and new_key not in params:
            params[new_key] = params.pop(old_key)
        elif old_key in params:
            params.pop(old_key)

    def _normalize_circle_detection_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        fit_modes = {
            "standardLSQ",
            "hyperLSQ",
            "riemannSWFLa",
            "prattSVD",
            "taubinSVD",
            "hyperSVD",
            "kmh",
        }
        contour_modes = {"RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE"}

        old_mode = params.get("mode")
        if old_mode in fit_modes and "circle_fit_mode" not in params:
            params["circle_fit_mode"] = old_mode
            params["mode"] = "RETR_EXTERNAL"
            self._node.get_logger().warn(
                "Converted CircleDetection.mode fit algorithm to circle_fit_mode."
            )
        elif old_mode not in contour_modes:
            params["mode"] = "RETR_EXTERNAL"

        if "circle_fit_mode" not in params:
            params["circle_fit_mode"] = "standardLSQ"
        if "fit_single_circle" not in params:
            params["fit_single_circle"] = False
        if "draw_circles" not in params:
            params["draw_circles"] = True

        return params

    def _tool_response(self, ok: bool, process_name: str, message: str, **kwargs) -> str:
        payload = {
            "ok": ok,
            "process_name": process_name,
            "message": message,
        }
        payload.update(kwargs)
        return json.dumps(payload, default=str)

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
                    function_key = "name"
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

    def check_pipeline_steps(self, pipeline_steps: List[Dict[str, Any]]) -> bool:
        """
        Checks if the pipeline steps are valid.
        The pipeline does not allow keys like 'function_name', 'function', or 'name' in the step dictionaries.
        """
        for step in pipeline_steps:
            if not isinstance(step, dict):
                return False
            for key in step.keys():
                if key in ['function_name', 'function', 'name']:
                    return False
        return True
