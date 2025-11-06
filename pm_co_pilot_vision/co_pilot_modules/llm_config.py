import os
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory

from langchain.chat_models import init_chat_model

class LLMConfig:

    def __init__(self, llm_tool: str):
        """
        Load LLM configuration from prompts.yaml. Accepts historical aliases:
        - 'agent' -> tries 'agent_all_functions' as default fallback
        """
        # Resolve prompts.yaml path robustly
        prompt_path = None
        try:
            share_dir = get_package_share_directory("pm_co_pilot_vision")
            # Preferred location at share root
            p1 = Path(share_dir) / "prompts.yaml"
            p2 = Path(share_dir) / "config" / "prompts.yaml"
            if p1.exists():
                prompt_path = p1
            elif p2.exists():
                prompt_path = p2
        except Exception:
            pass

        if prompt_path is None:
            # Fallback to source repo layout
            repo_p = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"
            if repo_p.exists():
                prompt_path = repo_p
            else:
                raise FileNotFoundError("prompts.yaml not found in share dir or local config/")

        with open(prompt_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file) or {}

        # Normalize alias and provide helpful fallback for 'agent'
        key = llm_tool
        if key not in config_data:
            if key == "agent":
                # Prefer FULL_SPECS as a sensible default if available
                if "agent_all_functions" in config_data:
                    key = "agent_all_functions"
                elif "agent_get_functions" in config_data:
                    key = "agent_get_functions"
            # As a final fallback, pick any key starting with 'agent'
            if key not in config_data:
                for k in config_data.keys():
                    if isinstance(k, str) and k.startswith("agent"):
                        key = k
                        break
        if key not in config_data:
            available = ", ".join(config_data.keys())
            raise KeyError(f"LLM config entry '{llm_tool}' not found. Available: {available}")

        entry = config_data[key]
        self.model = entry['model']
        self.model_provider = entry['model_provider']
        self.temperature = entry['temperature']
        self.system_prompt = entry['system_prompt']

        
        # self.llm = init_chat_model(self.model, model_provider=self.model_provider, temperature=self.temperature)