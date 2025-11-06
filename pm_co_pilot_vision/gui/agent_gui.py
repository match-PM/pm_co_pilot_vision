from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import traceback

# Qt imports (PyQt6)
try:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QLabel,
        QLineEdit,
        QTextEdit,
        QPushButton,
        QFileDialog,
        QComboBox,
        QGridLayout,
        QMessageBox,
    )
    from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
except Exception as e:  # pragma: no cover
    print("PyQt6 is required. Try: pip install PyQt6", file=sys.stderr)
    raise

import rclpy
from rclpy.node import Node

from pm_co_pilot_vision.co_pilot_modules.agent import Agent, FunctionsView
from pm_co_pilot_vision.utils.vision_functions import VisionHandler


class AgentWorker(QObject):
    result = pyqtSignal(str)
    error = pyqtSignal(str, str)  # message, details
    finished = pyqtSignal()

    def __init__(self, node: Node, prompt: str, image_name: str, view: FunctionsView, image_path: str, image_proc_dir: str):
        super().__init__()
        self._node = node
        self._prompt = prompt
        self._image_name = image_name
        self._view = view
        self._image_path = image_path
        self._image_proc_dir = image_proc_dir

    def run(self):  # executed in worker thread
        import traceback as _tb
        try:
            vh = VisionHandler(
                vision_node=self._node,
                image_name=self._image_name,
                image_path=self._image_path,
                image_processes_path=self._image_proc_dir,
            )
            agent = Agent(
                self._node,
                vision_handler=vh,
                functions_view=self._view,
            )
            response = agent.handle_user_input(self._prompt, image_name=self._image_name)
            self.result.emit(str(response))
        except Exception as e:
            msg = str(e) or e.__class__.__name__
            details = _tb.format_exc()
            try:
                self._node.get_logger().error(f"Agent worker error: {repr(e)}\n{details}")
            except Exception:
                pass
            self.error.emit(msg, details)
        finally:
            self.finished.emit()

class AgentWindow(QMainWindow):
    def __init__(self, node: Optional[Node] = None):
        super().__init__()
        self.setWindowTitle("PM Co-Pilot Vision - Agent GUI")
        self.resize(900, 650)

        # ROS2 init and node
        self._owns_rcl = False
        if node is not None:
            self.node = node
        else:
            if not rclpy.ok():
                rclpy.init()
                self._owns_rcl = True
            self.node: Node = rclpy.create_node("pm_co_pilot_vision_gui")

        # Widgets
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter user_prompt for the agent...")

        self.image_name_edit = QLineEdit()
        self.image_name_edit.setPlaceholderText("Enter image filename (e.g., sensor_corner.png)")

        self.view_combo = QComboBox()
        self.view_combo.addItem("Names only", FunctionsView.NAMES_ONLY)
        self.view_combo.addItem("Full specs", FunctionsView.FULL_SPECS)
        self.view_combo.setCurrentIndex(1)  # default to Full specs

        self.run_btn = QPushButton("Run Agent")
        self.run_btn.clicked.connect(self._run_agent)

        self.output_view = QTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setPlaceholderText("Agent response will appear here…")

        # Layout
        central = QWidget()
        layout = QGridLayout(central)
        row = 0
        layout.addWidget(QLabel("FunctionsView:"), row, 0)
        layout.addWidget(self.view_combo, row, 1, 1, 3)
        row += 1

        layout.addWidget(QLabel("Image name:"), row, 0)
        layout.addWidget(self.image_name_edit, row, 1, 1, 3)
        row += 1

        layout.addWidget(QLabel("User prompt:"), row, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.prompt_edit, row, 1, 1, 3)
        row += 1

        layout.addWidget(self.run_btn, row, 3, Qt.AlignmentFlag.AlignRight)
        row += 1

        layout.addWidget(QLabel("Response:"), row, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.output_view, row, 1, 1, 3)
        row += 1

        self.setCentralWidget(central)

    def closeEvent(self, event):  # noqa: N802
        try:
            if self._owns_rcl:
                if hasattr(self, "node") and self.node is not None:
                    self.node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
        finally:
            super().closeEvent(event)

    # --- UI actions ---
    # No file/directory browsing required in the simplified flow

    def _run_agent(self):
        prompt = self.prompt_edit.toPlainText().strip()
        image_name = self.image_name_edit.text().strip()
        view: FunctionsView = self.view_combo.currentData()  # type: ignore[assignment]

        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Please enter a user prompt.")
            return

        if not image_name:
            QMessageBox.warning(self, "Missing image name", "Please enter an image file name (e.g., sensor_corner.png).")
            return

        image_path = self._default_image_path()
        image_proc_dir = self._default_process_dir() or image_path

        # Validate image existence early to provide a clear error
        from pathlib import Path as _Path
        img_full = _Path(image_path) / image_name
        if not img_full.exists():
            hint = (
                f"Image not found: {img_full}\n\n"
                f"- Set PM_CO_PILOT_IMAGE_PATH to your images directory, or\n"
                f"- Place '{image_name}' under '{image_path}'."
            )
            QMessageBox.critical(self, "Image not found", hint)
            return

        # Run agent in a background thread to keep UI responsive
        self.output_view.append("Running agent…")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()

        # Keep references on self to avoid premature GC
        self._worker = AgentWorker(self.node, prompt, image_name, view, image_path, image_proc_dir)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        def on_result(text: str):
            self.output_view.append("\n=== Agent Response ===\n" + text)

        def on_error(msg: str, details: str):
            self._show_error("Agent error", msg)
            if details:
                # Also append details to the output for debugging
                self.output_view.append("\n[Details]\n" + details)

        def on_finished():
            self.run_btn.setEnabled(True)
            # Gracefully stop the worker thread
            try:
                self._thread.quit()
                self._thread.wait(2000)
            except Exception:
                pass
            # Cleanup refs
            self._worker = None
            self._thread = None

        # Connect signals so that worker.run executes in the worker thread
        self._thread.started.connect(self._worker.run)
        self._worker.result.connect(on_result)
        self._worker.error.connect(on_error)
        self._worker.finished.connect(on_finished)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _show_error(self, title: str, message: str):
        # Always show some text even if message is empty
        text = message if message else "An unknown error occurred."
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(text)
        box.exec()

    def _show_exception(self, title: str, exc: Exception):
        # Provide both a short message and expandable traceback details
        msg = str(exc) or exc.__class__.__name__
        details = traceback.format_exc()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(msg)
        if details:
            box.setDetailedText(details)
        box.exec()

    def _default_image_path(self) -> str:
        # Try environment variable first
        env_path = os.environ.get("PM_CO_PILOT_IMAGE_PATH")
        if env_path and Path(env_path).is_dir():
            return str(Path(env_path))
        # Fallback to typical project path used in your node example
        fallback = "/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_db/co_pilot_tests/"
        return fallback

    def _default_process_dir(self) -> Optional[str]:
        env_path = os.environ.get("PM_CO_PILOT_PROCESSES_PATH")
        if env_path and Path(env_path).is_dir():
            return str(Path(env_path))
        # Fallback to typical project path used in your node example
        fallback = "/home/match-mover/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_processes/co_pilot_tests"
        return fallback


def main():
    app = QApplication(sys.argv)
    w = AgentWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
