import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from PyQt5.QtWidgets import QApplication
from threading import Thread
import sys

from langchain.agents import AgentExecutor
from pm_co_pilot_vision.co_pilot_modules.agent import Agent



class PmCoPilotVisionNode(Node):

    def __init__(self):
        super().__init__('pm_co_pilot_vision_node')
        try:
            agent = Agent(self)
            agent_executor = agent.create_vision_pipeline_agent()
            messages = agent_executor.invoke({
                # "input": "Set a threshold of 128 and run the pipeline. Give the user a response about the result.",
                "input": "what funtions are available? Return the response",
                "chat_history": []
            })
            response = messages['output']
            self.get_logger().info(f"Agent response: {response}")
        except AttributeError as e:
            print(f"Caught AttributeError: {e}")
        except Exception as e:
            print(f"Caught general exception: {e}")

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=6) 

    # app = QApplication(sys.argv)

    co_pilot_vision_node = PmCoPilotVisionNode()
    executor.add_node(co_pilot_vision_node)

    thread = Thread(target=executor.spin)
    thread.start()
    
    try:
        co_pilot_vision_node.get_logger().info("PmCoPilotVisionNode is running...")
        # sys.exit(app.exec())

    finally:
        co_pilot_vision_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
