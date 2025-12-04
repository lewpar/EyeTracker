install dkms before hailo (sudo apt install dkms) // This is required for the pcie driver to properly install from the `hailo-all` package.
install hailo sdk and driver (sudo apt install hailo-all)
download and install miniforge (anaconda environment manager)
create python environment (conda create -n degirum python=3.12)
activate environment (conda activate degirum)
install degirum package (pip install degirum)
install pygame package (pip install pygame)
clone project (git clone https://github.com/lewpar/EyeTracker.git)

create separate folder (mkdir DegirumServer)
create zoo folder (mkdir ./DegirumServer/Zoo)
get inferencing models models (degirum download-zoo --url "https://hub.degirum.com/degirum/hailo" --path "./DegirumServer/Zoo") // Improve this instruction by downloading the correct model
start degirum server (degirum server)

run client (python main.py)