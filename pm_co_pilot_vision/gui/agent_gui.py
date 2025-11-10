from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, List
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
    QHBoxLayout,
        QVBoxLayout,
        QScrollArea,
        QMessageBox,
        QFrame,
    )
    from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer
    from PyQt6.QtGui import QPixmap
except Exception as e:  # pragma: no cover
    print("PyQt6 is required. Try: pip install PyQt6", file=sys.stderr)
    raise

import rclpy
from rclpy.node import Node
import yaml

from pm_co_pilot_vision.co_pilot_modules.agent import Agent, FunctionsView
from pm_co_pilot_vision.utils.vision_functions import VisionHandler


def load_available_models() -> List[str]:
    """Load available models from prompts.yaml."""
    import os
    from pathlib import Path

    # Try share dir first
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("pm_co_pilot_vision")
        prompt_path = Path(share_dir) / "prompts.yaml"
        if not prompt_path.exists():
            prompt_path = Path(share_dir) / "config" / "prompts.yaml"
    except Exception:
        pass

    if not prompt_path or not prompt_path.exists():
        # Fallback to repo
        repo_p = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"
        if repo_p.exists():
            prompt_path = repo_p

    if prompt_path and prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return data.get("available_models", [])
    return []


class AgentWorker(QObject):
    result = pyqtSignal(str)
    error = pyqtSignal(str, str)  # message, details
    processed_image_path = pyqtSignal(str)
    overlay_path_hint = pyqtSignal(str)  # emitted early so GUI can start watching
    finished = pyqtSignal()

    def __init__(self, node: Node, prompt: str, image_name: str, view: FunctionsView, image_path: str, image_proc_dir: str, model: Optional[str] = None):
        super().__init__()
        self._node = node
        self._prompt = prompt
        self._image_name = image_name
        self._view = view
        self._image_path = image_path
        self._image_proc_dir = image_proc_dir
        self._model = model

    def run(self):  # executed in worker thread
        import traceback as _tb
        try:
            vh = VisionHandler(
                vision_node=self._node,
                image_name=self._image_name,
                image_path=self._image_path,
                image_processes_path=self._image_proc_dir,
            )
            # tell GUI where the overlay image will be saved so it can watch it
            overlay_name = self._image_name.replace('.png', '_overlay.png')
            overlay_path = vh.image_result_path + overlay_name
            self.overlay_path_hint.emit(overlay_path)
            agent = Agent(
                self._node,
                vision_handler=vh,
                functions_view=self._view,
                model=self._model,
            )
            response = agent.handle_user_input(self._prompt, image_name=self._image_name)
            self.result.emit(str(response))
            # Emit the overlay image path for GUI display; fallback to processed
            overlay_name = self._image_name.replace('.png', '_overlay.png')
            overlay_path = vh.image_result_path + overlay_name
            if not os.path.exists(overlay_path):
                processed_name = self._image_name.replace('.png', '_processed.png')
                overlay_path = vh.image_result_path + processed_name
            self.processed_image_path.emit(overlay_path)
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

        self.model_combo = QComboBox()
        available_models = load_available_models()
        self.model_combo.addItems(available_models)
        if available_models:
            self.model_combo.setCurrentText(available_models[0])  # default to first

        self.run_btn = QPushButton("Run Agent")
        self.run_btn.clicked.connect(self._run_agent)

        self.output_view = QTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setPlaceholderText("Agent response will appear here…")

        # Right-side image area (vertical: original on top, overlay on bottom)
        self.original_image_label = QLabel("Original Image")
        # Border applied directly; no fixed size so we can adapt to image
        self.original_image_label.setStyleSheet("border: 1px solid #444; background: transparent;")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.overlay_image_label = QLabel("Overlay Image")
        self.overlay_image_label.setStyleSheet("border: 1px solid #444; background: transparent;")
        self.overlay_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Scroll areas allow large images while keeping window reasonable
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)
        self.original_scroll.setWidget(self.original_image_label)
        # Hide scrollbars; we'll scale images to fit instead
        self.original_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.original_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.original_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.overlay_scroll = QScrollArea()
        self.overlay_scroll.setWidgetResizable(True)
        self.overlay_scroll.setWidget(self.overlay_image_label)
        self.overlay_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overlay_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overlay_scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Layout restructure: horizontal split (left form, right images)
        central = QWidget()
        main_layout = QHBoxLayout(central)

        form_container = QWidget()
        form_layout = QGridLayout(form_container)
        form_container.setMaximumWidth(420)
        row = 0
        form_layout.addWidget(QLabel("FunctionsView:"), row, 0)
        form_layout.addWidget(self.view_combo, row, 1)
        row += 1
        form_layout.addWidget(QLabel("Model:"), row, 0)
        form_layout.addWidget(self.model_combo, row, 1)
        row += 1
        form_layout.addWidget(QLabel("Image name:"), row, 0)
        form_layout.addWidget(self.image_name_edit, row, 1)
        row += 1
        form_layout.addWidget(QLabel("User prompt:"), row, 0, Qt.AlignmentFlag.AlignTop)
        form_layout.addWidget(self.prompt_edit, row, 1)
        row += 1
        form_layout.addWidget(self.run_btn, row, 1, Qt.AlignmentFlag.AlignRight)
        row += 1
        form_layout.addWidget(QLabel("Response:"), row, 0, Qt.AlignmentFlag.AlignTop)
        form_layout.addWidget(self.output_view, row, 1)

        form_layout.setColumnStretch(0, 0)
        form_layout.setColumnStretch(1, 1)

        # Image panel
        image_panel = QVBoxLayout()
        image_panel.addWidget(QLabel("Original"))
        image_panel.addWidget(self.original_scroll)
        image_panel.addWidget(QLabel("Overlay"))
        image_panel.addWidget(self.overlay_scroll)

        main_layout.addWidget(form_container)
        # spacer/stretch between panels
        main_layout.addSpacing(8)
        images_container = QWidget()
        images_container.setLayout(image_panel)
        main_layout.addWidget(images_container, 1)

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
        model = self.model_combo.currentText() or None

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

        # Load original image
        self._load_original_image(image_path, image_name)

        # Run agent in a background thread to keep UI responsive
        self.output_view.append("Running agent…")
        self.run_btn.setEnabled(False)
        QApplication.processEvents()

        # Keep references on self to avoid premature GC
        self._worker = AgentWorker(self.node, prompt, image_name, view, image_path, image_proc_dir, model)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        def on_result(text: str):
            self.output_view.append("\n=== Agent Response ===\n" + text)

        def on_processed_image_path(path: str):
            self._load_overlay_image(path)
        def on_overlay_hint(path: str):
            # start polling timer to live-refresh overlay as soon as it appears
            self._overlay_target_path = path
            if not hasattr(self, '_overlay_timer'):
                self._overlay_timer = QTimer(self)
                self._overlay_timer.setInterval(750)  # ms
                self._overlay_timer.timeout.connect(self._poll_overlay_file)
            self._overlay_timer.start()

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
        self._worker.processed_image_path.connect(on_processed_image_path)
        self._worker.overlay_path_hint.connect(on_overlay_hint)
        self._worker.error.connect(on_error)
        self._worker.finished.connect(on_finished)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _load_original_image(self, image_path: str, image_name: str):
        full_path = Path(image_path) / image_name
        if full_path.exists():
            pixmap = QPixmap(str(full_path))
            if not pixmap.isNull():
                self._original_pixmap = pixmap
                self._update_label_scaled(self.original_image_label, self.original_scroll, pixmap)
            else:
                self.original_image_label.setText("Failed to load image")
        else:
            self.original_image_label.setText("Image not found")

    def _load_overlay_image(self, overlay_path: str):
        path = Path(overlay_path)
        if path.exists():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self._overlay_pixmap = pixmap
                self._update_label_scaled(self.overlay_image_label, self.overlay_scroll, pixmap)
            else:
                self.overlay_image_label.setText("Failed to load overlay image")
        else:
            self.overlay_image_label.setText("Overlay image not found yet")

    def _update_label_scaled(self, label: QLabel, scroll: QScrollArea, pixmap: QPixmap):
        """Scale pixmap to fit viewport; keep aspect; hide scrollbars by staying within viewport."""
        vp = scroll.viewport().size()
        # Nudge down a couple pixels to account for any internal margins
        target_w = max(1, vp.width() - 2)
        target_h = max(1, vp.height() - 2)
        if target_w > 1 and target_h > 1:
            scaled = pixmap.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            scaled = pixmap
        label.setPixmap(scaled)
        label.setFixedSize(scaled.size())

    def _poll_overlay_file(self):
        # periodically refresh overlay if present
        target = getattr(self, '_overlay_target_path', None)
        if not target:
            return
        p = Path(target)
        if p.exists():
            pixmap = QPixmap(str(p))
            if not pixmap.isNull():
                self._overlay_pixmap = pixmap
                self._update_label_scaled(self.overlay_image_label, self.overlay_scroll, pixmap)
        # stop timer if agent finished and overlay loaded once
        if not hasattr(self, '_worker') or self._worker is None:
            # give one last attempt then stop
            if hasattr(self, '_overlay_timer'):
                self._overlay_timer.stop()

    def resizeEvent(self, event):  # noqa: N802
        # On window resize, rescale images to fit their viewports
        try:
            self._rescale_images_to_fit()
        finally:
            return super().resizeEvent(event)

    def _rescale_images_to_fit(self):
        # Rescale currently loaded pixmaps to fit
        if hasattr(self, '_original_pixmap') and isinstance(self._original_pixmap, QPixmap) and not self._original_pixmap.isNull():
            self._update_label_scaled(self.original_image_label, self.original_scroll, self._original_pixmap)
        if hasattr(self, '_overlay_pixmap') and isinstance(self._overlay_pixmap, QPixmap) and not self._overlay_pixmap.isNull():
            self._update_label_scaled(self.overlay_image_label, self.overlay_scroll, self._overlay_pixmap)

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
