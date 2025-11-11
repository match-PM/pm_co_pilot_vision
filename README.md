# pm_co_pilot_vision

ROS 2 package that provides a Co‑Pilot Vision agent with a PyQt6 GUI. It lets you run a configurable vision pipeline on an image, visualize the live overlay produced by the pipeline, and interact with the agent via LLM-backed tools.

The package is designed to work both from an installed ROS share (installed with colcon) and directly from source during development. Key assets such as `prompts.yaml` and `vision_functions.json` are resolved from the package install path with a safe fallback to the local repository.

## Highlights

- PyQt6 GUI: enter the image name, select the tool view (FunctionsView), pick a model from `prompts.yaml`, and run the agent.
- Live overlay preview: the pipeline writes an overlay image during processing; the GUI watches and refreshes it automatically.
- Final vs overlay: the pipeline’s final output image is still saved and used by the agent, while the GUI displays the overlay so you can see intermediate annotations.
- ROS 2 node lifecycle: the GUI hosts a ROS2 node; it can be launched directly or from a ROS entry point.

## Repository layout

- `pm_co_pilot_vision/gui/agent_gui.py` — PyQt6 GUI (right pane shows Original on top and Overlay at the bottom).
- `pm_co_pilot_vision/pm_co_pilot_vision.py` — entry point that can launch the GUI inside a ROS2 node context.
- `pm_co_pilot_vision/co_pilot_modules/agent.py` — the Agent wrapper, with `FunctionsView` enum and optional model override.
- `pm_co_pilot_vision/utils/vision_functions.py` — VisionHandler that orchestrates the vision pipeline and file outputs.
- `config/prompts.yaml` — models and prompt configuration; the GUI reads available models from here.
- `files/vision_functions.json` — vision tool/function specs used by the agent.
- `launch/pm_co_pilot_vision.launch.py` — example launch file.

## Requirements

- ROS 2 (tested with humble)
- Python 3.10+
- PyQt6
- Project dependencies that the package imports at runtime:
	- `pm_vision_manager` (pipeline, camera configs)

You can install `pm_vision_manager` by cloning its repository into your ROS 2 workspace and building it with colcon.

Install Python user deps (PyQt6) into the environment you use to run ROS:

```bash
pip install --user PyQt6
```

If you’re using a venv, activate it first; if you’re using the ROS Python, consider creating a venv to avoid mixing system packages.

## Build

Place the package in your ROS 2 workspace and build with colcon:

```bash
cd ~/ros2_ws/src
git clone <this-repo-url> pm_co_pilot_vision
cd ..
colcon build --packages-select pm_co_pilot_vision
source install/setup.bash
```

## Run

You can run the GUI via the package executable or with a launch file (if wired in your environment):

```bash
# direct executable (installed via entry_points)
ros2 run pm_co_pilot_vision pm_co_pilot_vision_gui

# or, if you prefer using the launch file (example)
ros2 launch pm_co_pilot_vision pm_co_pilot_vision.launch.py
```

### Environment variables (image & processes paths)

At runtime the GUI expects only the image file name (e.g., `sensor_corner.png`). It locates paths as follows:

- `PM_CO_PILOT_IMAGE_PATH` (optional): directory containing your input images.
- `PM_CO_PILOT_PROCESSES_PATH` (optional): directory where pipeline JSON files will be written.

If these variables are not set, the code uses sensible fallbacks that match typical `pm_vision_manager` locations, e.g.:

- Images default: `/home/<user>/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_db/co_pilot_tests/`
- Processes default: `/home/<user>/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_processes/co_pilot_tests`

Exporting them before launching is recommended:

```bash
export PM_CO_PILOT_IMAGE_PATH=/path/to/images
export PM_CO_PILOT_PROCESSES_PATH=/path/to/vision_processes
```

## Using the GUI

1. Start the app: `ros2 run pm_co_pilot_vision pm_co_pilot_vision_gui`.
2. Fill the fields on the left:
	 - FunctionsView: Names only or Full specs
	 - Model: comes from `config/prompts.yaml` (see below)
	 - Image name: just the filename (e.g., `sensor_corner.png`) located in `PM_CO_PILOT_IMAGE_PATH`
	 - User prompt: free text prompt for the agent
3. Click “Run Agent”.
4. Right pane shows two images:
	 - Original (top)
	 - Overlay (bottom): this updates live as the pipeline saves the overlay file.

Outputs:
- Final processed image is saved as `<image>_processed.png` under an auto-created result directory.
- Live overlay is saved as `<image>_overlay.png` in the same directory; the GUI watches it and refreshes automatically.
- A results JSON (`vision_results.json`) is also written with serializable content extracted from the pipeline/agent.

## Configuration files

### prompts.yaml

The GUI reads “available_models” from `prompts.yaml` to populate the model dropdown. The file is resolved in this order:

1. Package share directory: `$AMENT_PREFIX/share/pm_co_pilot_vision/prompts.yaml` or `$AMENT_PREFIX/share/pm_co_pilot_vision/config/prompts.yaml`.
2. Fallback to local repo: `config/prompts.yaml`.

You can add a new model to the dropdown by editing `config/prompts.yaml`:

```yaml
available_models:
	- gpt-5
	- gpt-5-mini
	- any other model available with langchain
```

### vision_functions.json

Function/tool specifications for the agent. Resolved in this order:

1. Package share directory: `$AMENT_PREFIX/share/pm_co_pilot_vision/vision_functions.json`
2. Fallback to local repo: `files/vision_functions.json`

## Architecture (quick tour)

- `Agent` (`co_pilot_modules/agent.py`): wraps the LLM, accepts `functions_view: FunctionsView` and optional `model` override.
- `VisionHandler` (`utils/vision_functions.py`): interfaces with `pm_vision_manager` to run the pipeline, writes output files, builds serializable results.
- GUI (`gui/agent_gui.py`):
	- Runs the agent work on a `QThread` with a `QObject` worker to keep the UI responsive.
	- Emits a path hint for the overlay early so the GUI can start watching and updating live.
	- Scales images to fit the viewport; borders hug the image contents (no excess frames). Scrollbars are disabled.

## Troubleshooting

- PyQt6 isn’t found
	- Install it in your runtime Python: `pip install --user PyQt6` and ensure you run the GUI in that environment.

- Image not found dialog
	- Set `PM_CO_PILOT_IMAGE_PATH` or place the image file under the default path shown in the dialog.

- NameError: `QObject` is not defined
	- Ensure the GUI imports include `from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread`.

- KeyError: `agent` in `prompts.yaml`
	- The loader now aliases `'agent'` to `'agent_all_functions'`. Make sure your `prompts.yaml` structure matches the examples or use the updated keys.

- GUI freezes while running
	- The agent runs in a worker `QThread`. If you changed that code and see freezes, verify the long-running calls are off the main thread.

- Scrollbars on the image panel
	- The GUI scales images to the viewport and hides scrollbars. If the window is too small, enlarge it or resize the panes.

## Development

- Build fast:

```bash
colcon build --packages-select pm_co_pilot_vision
source install/setup.bash
```




