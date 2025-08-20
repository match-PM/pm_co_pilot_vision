from rclpy.node import Node

from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
import operator, json
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, FunctionMessage

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
import os
import cv2
import base64, mimetypes
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


class AgentState(TypedDict):
    # Chat history the prebuilt ReAct agent expects:
    messages: Annotated[Sequence[BaseMessage], operator.add]
    goal: str
    image_path: str
    image_facts: Dict[str, Any]

    best_run: Optional[Dict[str, Any]]
    iter: int
    max_iters: int

    # anti-stuck helpers
    param_call_streak: int
    seen_schemas: Dict[str, Any]   # function_name -> schema/json

def _extract_last_pipeline_result(messages: Sequence[BaseMessage]) -> Optional[Dict[str, Any]]:
    """
    Look for the newest tool output that *looks* like a pipeline execution result.
    We scan ToolMessage/FunctionMessage content (tool OUTPUT), not tool_call args.
    """
    for msg in reversed(messages):
        if isinstance(msg, (ToolMessage, FunctionMessage)):
            data = None
            if isinstance(msg.content, str):
                try:
                    data = json.loads(msg.content)
                except Exception:
                    data = None
            elif isinstance(msg.content, dict):
                data = msg.content
            if isinstance(data, dict) and (
                "detections" in data or "metrics" in data or "score" in data or "passed" in data
            ):
                return data
    return None


def _last_tool_name(messages: Sequence[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.name  # name is the tool's name that produced this output
    return None


def _last_ai_text(messages: Sequence[BaseMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


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

        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}    

        self.vision_functions = vision_handler
        self.available_functions = self.vision_functions.return_available_function_names()


        # Inject available functions into system prompt
        self.system_prompt = raw_system_prompt.format(
            available_functions=self.available_functions
        )

        # self.node.get_logger().info(f"System prompt: {self.system_prompt}")

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            MessagesPlaceholder(variable_name="messages") # Keep this to include conversation history
        ])        

        self.vision_tools = Tools(node, vision_handler=vision_handler)

        try:
            self.tools = [
                VisionPipelineTool(node, vision_handler=vision_handler),
                self.vision_tools.get_function_parameters_tool
            ]
        except Exception as e:
            self.node.get_logger().error(f"Error initializing tools: {e}")    

        self._build_graph()

        
    def _build_graph(self):
        """
        agent_step -> evaluate -> (continue | end)
        """
        agent_runnable = create_react_agent(
                self.model,
                self.tools,
                checkpointer=self.memory,
                prompt=self.prompt
            )

        def agent_step(state: AgentState) -> AgentState:
            """Run one turn of the ReAct agent using messages provided in state."""
            result = agent_runnable.invoke({"messages": state["messages"]}, self.config)
            # The prebuilt returns {"messages": [...]} including new AI/tool messages
            state["messages"] = result["messages"]
            return state

        def evaluate(state: AgentState) -> AgentState:
            """
            Inspect the most recent tool output (if any), compute/update a simple score,
            keep track of the best run, and increment the iteration counter.
            """
            ai_text = _last_ai_text(state["messages"])
            if ai_text:
                self.node.get_logger().info(f"[AI] {ai_text}")

            # Track which tool ran last (for anti-stuck)
            last_tool = _last_tool_name(state["messages"])
            if last_tool == "get_function_parameters":
                state["param_call_streak"] = state.get("param_call_streak", 0) + 1
            elif last_tool:
                state["param_call_streak"] = 0  # reset when any *other* tool runs

            self.node.get_logger().info(f"Last tool called: {last_tool}", param_call_streak=state.get("param_call_streak", 0))

            # Pull any pipeline result and track best
            out = _extract_last_pipeline_result(state["messages"])
            if out:
                score = float(out.get("score", 0.0))
                if not score:
                    # cheap heuristic if your tool doesn't provide a score yet
                    det = out.get("detections", {})
                    cnt = det.get("count") if isinstance(det, dict) else None
                    if cnt is None and isinstance(det, dict):
                        for k in ("corners", "circles", "lines", "blobs"):
                            if k in det and isinstance(det[k], list):
                                cnt = max(cnt or 0, len(det[k]))
                    score = 1.0 if (cnt or 0) > 0 else 0.0
                out["__score"] = score

                best = state.get("best_run")
                if best is None or out["__score"] > best.get("__score", -1):
                    state["best_run"] = out
            
                        # Nudge the agent if it's stuck calling the params tool
            if state.get("param_call_streak", 0) >= 3:
                hint = (
                    "You have already retrieved the needed function parameters. "
                    "Do not call `get_function_parameters` again unless a NEW function appears. "
                    "Now propose a pipeline and execute it with `VisionPipelineTool`."
                )
                state["messages"] = list(state["messages"]) + [HumanMessage(content=hint)]
                # Reset streak so we don't keep injecting
                state["param_call_streak"] = 0

            # IMPORTANT: always advance the loop counter to avoid infinite loops
            state["iter"] = int(state.get("iter", 0)) + 1
            return state

        def should_continue(state: AgentState) -> str:
            best = state.get("best_run") or {}
            passed = bool(best.get("passed", False))
            good = float(best.get("__score", 0.0)) >= 0.9  # tune as you like
            budget = state["iter"] >= state["max_iters"]
            return "end" if budget else "continue"

        # Assemble the graph
        g = StateGraph(AgentState)
        g.add_node("agent_step", agent_step)
        g.add_node("evaluate", evaluate)

        g.add_edge(START, "agent_step")
        g.add_edge("agent_step", "evaluate")
        g.add_conditional_edges("evaluate", should_continue, {"continue": "agent_step", "end": END})

        # Compile with the same checkpointer
        self.app = g.compile(checkpointer=self.memory)


    def handle_user_input(self, user_message: str, image_name: str) -> str:
        """
        Kick off the LangGraph loop with:
          - one HumanMessage containing: user text + image + concise image facts
          - small iteration budget for refine loops
        Returns the latest AI response if present, otherwise a compact summary of best_run.
        """
        self.node.get_logger().info(f"Handling user input: {user_message}")

        # Validate image
        if not image_name:
            self.node.get_logger().error("Valid image name is required for processing.")
            return "Valid image name is required for processing."

        # Build multimodal content blocks
        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]

        data_url = self.vision_functions.path_to_data_url(image_name)
        if not (isinstance(data_url, str) and data_url.startswith("data:")):
            self.node.get_logger().error("Failed to generate valid data URL for image.")
            return "Failed to process image for agent."
        content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

        # Compute concise image facts once
        image_facts = self.vision_functions.compute_image_facts(image_name)
        content_blocks.append({"type": "text", "text": f"Image facts (concise): {image_facts}"})


        # Seed conversation + state
        init_state: AgentState = {
            "messages": [HumanMessage(content=content_blocks)],
            "goal": user_message,
            "image_name": image_name,
            "image_facts": image_facts,
            "best_run": None,
            "iter": 0,
            "max_iters": 5,             # tweak as needed
            "param_call_streak": 0,
            "seen_schemas": {},         # filled as agent fetches schemas
        }

        # temporary, inside handle_user_input after building init_state
        for event in self.app.stream(init_state, self.config):
            # event is a dict like {"agent_step": {...}} or {"evaluate": {...}}
            self.node.get_logger().info(f"[GRAPH EVENT] {list(event.keys())[0]}")
        final_state = self.app.get_state(self.config)  # end state after stream

        # Run the LangGraph loop
        # final_state: AgentState = self.app.invoke(init_state, self.config)

        # Prefer the last AIMessage content
        ai_msgs = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
        if ai_msgs:
            self.node.get_logger().info("Returning final AI message content.")
            return ai_msgs[-1].content

        # Fallback: return the best_run JSON
        best = final_state.get("best_run")
        if best:
            self.node.get_logger().info("Returning best_run summary.")
            return f"Best run: {json.dumps(best)}"

        self.node.get_logger().warn("No result produced.")
        return "No result produced."
    

    
    

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

