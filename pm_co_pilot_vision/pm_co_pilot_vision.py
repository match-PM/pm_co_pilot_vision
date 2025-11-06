import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from PyQt6.QtWidgets import QApplication
from threading import Thread
import sys

from pm_co_pilot_vision.gui.agent_gui import AgentWindow


class PmCoPilotVisionNode(Node):
    """Legacy node kept for compatibility (not used by GUI startup)."""
    def __init__(self):
        super().__init__('pm_co_pilot_vision_node')



def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create a lightweight node for the GUI to use
    gui_node = rclpy.create_node('pm_co_pilot_vision_gui_node')
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(gui_node)

    # Spin ROS in a background thread
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Start Qt application and show GUI bound to the existing ROS node
    app = QApplication(sys.argv)
    window = AgentWindow(node=gui_node)
    window.show()

    try:
        app.exec()
    finally:
        # Shutdown ROS nicely when the GUI exits
        executor.shutdown()
        if gui_node is not None:
            gui_node.destroy_node()
        rclpy.shutdown()


# def main(args=None):
#     rclpy.init(args=args)
#     executor = MultiThreadedExecutor(num_threads=6) 

#     # app = QApplication(sys.argv)

#     co_pilot_vision_node = PmCoPilotVisionNode()
#     executor.add_node(co_pilot_vision_node)

#     thread = Thread(target=executor.spin)
#     thread.start()
    
#     try:
#         co_pilot_vision_node.get_logger().info("PmCoPilotVisionNode is running...")
#         # sys.exit(app.exec())

#     finally:
#         co_pilot_vision_node.destroy_node()
#         executor.shutdown()
#         rclpy.shutdown()

if __name__ == '__main__':
    main()
