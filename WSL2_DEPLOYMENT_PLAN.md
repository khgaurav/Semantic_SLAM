# WSL2 Deployment Plan

This plan sets up a Windows machine with an NVIDIA RTX 4070 to run `Semantic_SLAM` localization inference inside WSL2, then replay or analyze the results either on that machine or back on the main workstation.

## Goal

Use the Windows/4070 machine for:

- headless localization inference
- faster SigLIP embedding computation
- result recording

Keep the current machine for:

- map building if desired
- report plots
- replay / visualization
- iterative analysis

## Recommended Architecture

```text
Windows 11 + WSL2 + Ubuntu 22.04
  -> ROS2 Humble
  -> CUDA-enabled PyTorch
  -> this repo
  -> KAIST bags + hybrid map
  -> headless localization runs

Main workstation
  -> receives recorded eval run directory
  -> replays / visualizes / compares results
```

## Assumptions

- Windows machine is on the same local network.
- Windows machine has an NVIDIA RTX 4070 and current NVIDIA drivers.
- You will use `WSL2` with Ubuntu `22.04`.
- The repo state on the GPU machine should match this repo closely enough to run the same scripts.
- The bags and map can either be copied directly or accessed from a shared folder.

## Phase 1: Windows + WSL2 Base Setup

### 1. Install WSL2 and Ubuntu 22.04

Run in an elevated PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted.

### 2. Confirm GPU is visible inside WSL

Inside Ubuntu:

```bash
nvidia-smi
```

Expected result: the RTX 4070 should appear.

If `nvidia-smi` is missing or fails:

- update the Windows NVIDIA driver first
- update WSL:

```powershell
wsl --update
```

## Phase 2: Ubuntu / ROS2 / Build Tooling

### 1. Update apt and install basics

Inside WSL Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  curl \
  git \
  python3-pip \
  python3-venv \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3-opencv \
  tmux
```

### 2. Install ROS2 Humble

Follow the normal Ubuntu 22.04 ROS2 Humble steps. In short:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt-get update
sudo apt-get install -y ros-humble-desktop ros-dev-tools
```

Initialize rosdep:

```bash
sudo rosdep init
rosdep update
```

### 3. Add ROS to shell startup

Append to `~/.bashrc`:

```bash
source /opt/ros/humble/setup.bash
```

Reload:

```bash
source ~/.bashrc
```

## Phase 3: Python Environment for GPU Inference

### 1. Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

### 2. Install CUDA-enabled PyTorch

Use the PyTorch wheel appropriate for the current CUDA-supported WSL setup. A typical command is:

```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install project Python dependencies

From the repo root:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install faiss-cpu rosbags
```

Notes:

- `faiss-cpu` is fine here unless you explicitly want FAISS GPU work.
- The main acceleration win is SigLIP inference on the 4070.

### 4. Verify PyTorch sees CUDA

```bash
python3 - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name =", torch.cuda.get_device_name(0))
PY
```

Expected result:

```text
cuda_available = True
device_name = NVIDIA GeForce RTX 4070
```

## Phase 4: Copy Repo + Data

## Option A: Copy everything locally to the WSL machine

Recommended for best performance.

Copy:

- repo source
- `data/kaist_ros2/urban39/`
- `data/hybrid_maps/urban38/`
- optionally `data/huggingface/` to avoid model redownload

Minimum required runtime files:

```text
Semantic_SLAM/
  scripts/
  src/
  requirements.txt
  rviz/
  install/            (not required if rebuilding there)
  data/
    kaist_ros2/
      urban39/
    hybrid_maps/
      urban38/
        map_index.faiss
        keyframe_poses.npy
        keyframe_ids.npy
        keyframe_stamps.npy
        keyframe_previews/
```

If copying from Linux to Windows over the network, `rsync` over SSH is ideal if available.

## Option B: Use a network share

Possible, but less ideal.

Use this only if copying the bags is inconvenient. Bag playback over a network share may be slower or less predictable than local disk.

## Phase 5: Build the Workspace on WSL

From the repo root:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select hybrid_localization
source install/setup.bash
```

## Phase 6: Warm the Model Cache

Before running a full bag, make sure Hugging Face model weights can load once.

Optional:

```bash
python3 - <<'PY'
from transformers import AutoModel, AutoProcessor
model_name = "google/siglip-base-patch16-224"
AutoProcessor.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
print("SigLIP cache ready")
PY
```

If you copied the local `data/huggingface/` cache from the Linux machine, you can point to it with:

```bash
export HF_HOME=$PWD/data/huggingface
```

## Phase 7: Run Headless Localization on the 4070 Machine

This is the main command:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
scripts/run_kaist_localization_headless.sh --map-sequence urban38 urban39
```

What it does:

- runs localization without RViz or `rqt`
- records `/localized_pose`
- records `/localized_path`
- records diagnostics and reference odometry
- writes an eval run directory under:

```text
data/eval/urban39_from_urban38/<timestamp>/
```

### Monitor progress

Use:

```bash
tail -f data/eval/urban39_from_urban38/<timestamp>/localization.log
```

Or:

```bash
tail -f \
  data/eval/urban39_from_urban38/<timestamp>/localization.log \
  data/eval/urban39_from_urban38/<timestamp>/record.log
```

## Phase 8: Bring Results Back

Copy the completed run directory back to the main machine:

```text
data/eval/urban39_from_urban38/<timestamp>/
```

Important files inside:

```text
pose_error.csv
pose_error_summary.json
record/
eval_env.txt
localization.log
```

## Phase 9: Replay Fast on the Main Machine

Once the completed run directory is copied back:

```bash
scripts/replay_kaist_localization_results.sh --rate 10.0 data/eval/urban39_from_urban38/<timestamp>
```

That replays:

- the original camera bag
- the recorded `/localized_pose`
- the recorded `/localized_path`

without recomputing inference.

## Optional: Run Replay on the Windows Machine Instead

If RViz / camera replay is also desired on the WSL machine:

```bash
scripts/replay_kaist_localization_results.sh --rate 10.0 data/eval/urban39_from_urban38/<timestamp>
```

This is less important if the main purpose of the Windows machine is only inference acceleration.

## Recommended First Test

Before a full bag run, verify CUDA inference is actually used:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

Then do one full headless run:

```bash
scripts/run_kaist_localization_headless.sh --map-sequence urban38 urban39
```

Check the node log for:

```text
Using device: cuda
```

## Common Failure Points

### `torch.cuda.is_available()` is false

Usually:

- Windows NVIDIA driver is too old
- WSL was not updated
- wrong PyTorch wheel installed

### ROS2 builds, but script cannot find packages

Usually:

- forgot `source /opt/ros/humble/setup.bash`
- forgot `source install/setup.bash`

### Model redownload is slow

Copy `data/huggingface/` from the current machine, or let the 4070 box download it once.

### Run is still CPU-bound

Check the localization log for:

```text
Using device: cuda
```

If it says `cpu`, the GPU setup is not active yet.

## Suggested Transfer Checklist

Copy these first:

```text
repo root
data/kaist_ros2/urban39/
data/hybrid_maps/urban38/
data/huggingface/   (optional but helpful)
```

## Success Criteria

The deployment is successful if:

1. `nvidia-smi` works inside WSL
2. `torch.cuda.is_available()` returns `True`
3. `colcon build` succeeds
4. `run_kaist_localization_headless.sh --map-sequence urban38 urban39` completes
5. the resulting `pose_error_summary.json` is written
6. replay works using `replay_kaist_localization_results.sh`

## Exact First Commands to Try

From a fresh WSL shell after setup:

```bash
cd ~/Semantic_SLAM
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select hybrid_localization
source install/setup.bash
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
scripts/run_kaist_localization_headless.sh --map-sequence urban38 urban39
```

