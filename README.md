# Eye Tracker
Eye Tracker is a PyGame rendered eyeball that tracks a targets face using image inference using the Hailo module for the Raspberry Pi 5.

Your mileage may vary for other devices.

## Setup
1. Install DKMS (required for the PCIE driver) and Hailo SDK + Driver
```
sudo apt install -y dkms && sudo apt install -y hailo-all
``` 
2. Download and install [Miniforge](https://github.com/conda-forge/miniforge) to manage Python environments.
3. Create and activate a virtual environment
```
conda create --name eyetracker python=3.12
conda activate eyetracker
```
4. Install dependencies
```
pip install degirum pygame
```
5. Get Degirum Hailo Model Zoo
```
degirum download-zoo --url "https://hub.degirum.com/degirum/hailo" --path "./zoo"
```
6. Clone and Run Eye Tracker
```
git clone https://github.com/lewpar/EyeTracker.git
python main.py
```

## Dependencies
- Degirum SDK - Used to interface with the Hailo AI accelerator
- PyGame - Used to render the eye