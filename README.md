# pm_co_pilot_vision

This package provides a vision co-pilot agent for ROS 2, enabling users to build opencv based image processing pipelines to detect 2D-features. 
It supports both a graphical user interface (GUI) for manual image selection and prompt entry.


## Repository layout

- `pm_co_pilot_vision/pm_co_pilot_vision.py` — main entry point with mode selection (GUI or node)
- `pm_co_pilot_vision/gui/agent_gui.py` — PyQt6 GUI with image browser and live preview
- `pm_co_pilot_vision/co_pilot_modules/agent.py` — Agent wrapper with `FunctionsView` enum and model override
- `pm_co_pilot_vision/utils/vision_functions.py` — VisionHandler for pipeline orchestration and file outputs
- `config/prompts.yaml` — models and prompt configuration
- `config/node_config.yaml` — settings for node mode (image path, prompt, etc.)
- `config/gui_config.yaml` — persistent GUI settings (last used directory)
- `files/vision_functions.json` — vision tool/function specifications
- `launch/pm_co_pilot_vision.launch.py` — launch file with mode argument

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

For hosted Hugging Face models through LangChain, install the Hugging Face integration and expose your token:

```bash
pip install --user langchain-huggingface huggingface_hub
export HUGGINGFACEHUB_API_TOKEN=hf_...
```


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

The package supports two modes: **GUI mode** (default) for interactive use and **Node mode** for automated execution.

### GUI Mode (Interactive)

Launch the GUI for manual image selection, prompt entry, and live preview:

```bash
# Using ros2 launch (recommended)
ros2 launch pm_co_pilot_vision pm_co_pilot_vision.launch.py

# or explicitly specify GUI mode
ros2 launch pm_co_pilot_vision pm_co_pilot_vision.launch.py mode:=gui

# Using ros2 run directly
ros2 run pm_co_pilot_vision pm_co_pilot_vision
```

### Node Mode (Headless/Automated)

Run the agent with hardcoded settings from `config/node_config.yaml`:

```bash
ros2 launch pm_co_pilot_vision pm_co_pilot_vision.launch.py mode:=node
```

This mode loads configuration from `node_config.yaml` including:
- Image path and name
- User prompt
- Processing directory

Perfect for automated workflows, testing, or integration with other ROS nodes.

### Environment Variables (GUI Mode)

When using GUI mode, you can optionally set these environment variables to customize default paths:

- `PM_CO_PILOT_IMAGE_PATH`: directory containing your input images
- `PM_CO_PILOT_PROCESSES_PATH`: directory where pipeline JSON files will be written

If not set, the GUI uses fallback defaults that match typical `pm_vision_manager` locations:

- Images default: `/home/<user>/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_db/co_pilot_tests/`
- Processes default: `/home/<user>/Documents/ros2_ws/src/pm_vision_manager/pm_vision_manager/vision_processes/co_pilot_tests`

**Note**: When you browse for an image in the GUI, the selected directory is automatically saved to `gui_config.yaml` and used for subsequent operations, overriding these defaults.

Example:

```bash
export PM_CO_PILOT_IMAGE_PATH=/path/to/images
export PM_CO_PILOT_PROCESSES_PATH=/path/to/vision_processes
ros2 launch pm_co_pilot_vision pm_co_pilot_vision.launch.py
```

## Using the GUI

1. **Start the app**: 
   ```bash
   ros2 launch pm_co_pilot_vision pm_co_pilot_vision.launch.py
   ```

2. **Select an image**:
   - **Option A (Browse)**: Click the "Browse..." button next to Image name field
     - A file dialog opens to your last used directory (or default)
     - Select an image file (PNG, JPG, JPEG, BMP, TIFF)
     - Image name is auto-filled and the image appears in the "Original" preview
     - Directory is saved to `gui_config.yaml` for next time
   - **Option B (Manual)**: Type the filename directly (e.g., `sensor_corner.png`)
     - Image must exist in `PM_CO_PILOT_IMAGE_PATH` or last browsed directory

3. **Configure agent settings**:
   - **FunctionsView**: Choose "Names only" or "Full specs"
   - **Model**: Select from dropdown (populated from `prompts.yaml`)
   - **User prompt**: Enter free text instruction for the agent

4. **Run**: Click "Run Agent"

5. **View results**:
   - **Left panel**: Agent response appears in the text area
   - **Right panel** (Preview):
     - **Original** (top): Selected input image
     - **Overlay** (bottom): Live-updating overlay with annotations
       - Updates automatically as the pipeline processes

**Outputs**:
- Final processed image: `<image>_processed.png`
- Live overlay image: `<image>_overlay.png` (shown in GUI)
- Results JSON: `vision_results.json` with pipeline metadata
- All saved to auto-created result directory

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
	- huggingface:cohere:CohereLabs/aya-vision-32b
	- huggingface:zai-org:zai-org/GLM-4.5V
	- any other model available with langchain
```

Models prefixed with `huggingface:` are loaded as hosted Hugging Face endpoints through `ChatHuggingFace(HuggingFaceEndpoint(...))`. Use `huggingface:<provider>:<model_id>` when the model requires a specific Inference Provider, for example `huggingface:cohere:CohereLabs/aya-vision-32b`. Use `huggingface:auto:<model_id>` to let Hugging Face choose among enabled providers. The default Hugging Face task is `image-text-to-text`, so select vision-language models that support image URL or data-URL inputs.

Only add a Hugging Face model to this list after a direct `InferenceClient.chat_completion(...)` image test works with your account and provider settings.

### vision_functions.json

Function/tool specifications for the agent. Resolved in this order:

1. Package share directory: `$AMENT_PREFIX/share/pm_co_pilot_vision/vision_functions.json`
2. Fallback to local repo: `files/vision_functions.json`

## Architecture

- **Main Entry** (`pm_co_pilot_vision.py`):
  - Parses `--mode` argument (gui or node)
  - GUI mode: launches PyQt6 interface with ROS node in background thread
  - Node mode: executes agent with settings from `node_config.yaml`
  
- **Agent** (`co_pilot_modules/agent.py`):
  - Wraps the LLM with `functions_view: FunctionsView` and optional `model` override
  - Supports both compact (names only) and detailed (full specs) function exposure
  
- **VisionHandler** (`utils/vision_functions.py`):
  - Interfaces with `pm_vision_manager` to run the vision pipeline
  - Writes output files (processed, overlay, results JSON)
  - Builds serializable results for the agent
  

## Troubleshooting

### Installation & Dependencies

- **PyQt6 isn't found**
  - Install it in your runtime Python: `pip install --user PyQt6` and ensure you run the GUI in that environment

- **NameError: `QObject` is not defined**
  - Ensure the GUI imports include `from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread`

- **pm_vision_manager not found**
  - Clone and build `pm_vision_manager` in your ROS 2 workspace
  - Source the workspace: `source ~/ros2_ws/install/setup.bash`

### Configuration Issues

- **KeyError: `agent` in `prompts.yaml`**
  - The loader aliases `'agent'` to `'agent_all_functions'`
  - Ensure your `prompts.yaml` has the correct structure (see Configuration files section)

- **Config file not found (node_config.yaml)**
  - Verify the file exists in `config/node_config.yaml`
  - Check that the package was built and installed correctly
  - The file should be in the share directory: `$ROS_INSTALL_PATH/share/pm_co_pilot_vision/config/`

### GUI Issues

- **Image not found dialog**
  - **Option 1**: Use the Browse button to select the image directly
  - **Option 2**: Set `PM_CO_PILOT_IMAGE_PATH` environment variable
  - **Option 3**: Place the image file under the default path shown in the dialog

- **Browse button doesn't save directory**
  - Check write permissions for `config/gui_config.yaml`
  - Verify the config directory exists and is writable

- **GUI freezes while running**
  - The agent runs in a worker `QThread` - if you modified that code, verify long-running calls are off the main thread

- **Scrollbars on the image panel**
  - The GUI scales images to the viewport and hides scrollbars
  - If the window is too small, enlarge it or resize the panes using the splitter

- **Overlay image not updating**
  - Check that the vision pipeline is saving `<image>_overlay.png`
  - Verify write permissions in the image_processes_path directory
  - The GUI polls every 750ms - ensure the file is being created

### Node Mode Issues

- **Node mode doesn't run**
  - Verify `config/node_config.yaml` exists and has correct structure
  - Check all paths in the config file are valid and accessible
  - Ensure the image file exists at the specified path

- **Launch argument not recognized**
  - Make sure you're using the correct syntax: `mode:=node` (not `--mode node`)
  - Verify the launch file was installed: `ros2 launch pm_co_pilot_vision pm_co_pilot_vision.launch.py --show-args`

## Development

- Build fast:

```bash
colcon build --packages-select pm_co_pilot_vision
source install/setup.bash
```
