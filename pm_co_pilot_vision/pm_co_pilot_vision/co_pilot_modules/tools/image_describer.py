
import json
from typing import ClassVar
from pydantic import PrivateAttr
from langchain.tools import BaseTool
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from pm_co_pilot_vision.co_pilot_modules.llm_config import LLMConfig
from pm_co_pilot_vision.utils.vision_functions import VisionHandler

from langchain.chat_models import init_chat_model

class ImageDescriberTool(BaseTool):
    name: ClassVar[str] = "image_describer"
    description: ClassVar[str] = ""
    
    _raw_doc: dict = PrivateAttr()
    _node: Node = PrivateAttr()

    def __init__(self, node: Node, vision_handler: VisionHandler):
        super().__init__()
        self._node = node
        self._vision_handler = vision_handler

    def _run(self, user_request: str, image_name: str) -> str:
        """Call the VLM to describe the image based on user request."""
        # Load the LLM configuration
        llm_config = LLMConfig("image_describer")

        prompt = llm_config.system_prompt
        model = llm_config.model
        model_provider = llm_config.model_provider
        temperature = llm_config.temperature

        image_facts = self._vision_handler.compute_image_facts(image_name)
        image_data_url = self._vision_handler.path_to_data_url(image_name, greyscale=True, resize=True)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_request},
            {"role": "user", "content": json.dumps(image_facts)},
            {"role": "user", "content": image_data_url}
        ]

        chat = init_chat_model(model=model, temperature=temperature)

        response = chat.invoke(messages)

        self._node.get_logger().info(f"ImageDescriberTool response: {response}")

        return response.content



