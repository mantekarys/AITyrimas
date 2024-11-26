# CARLA simulator experiments

[Gym wrapper](https://discord.com/channels/1283464486627315815/1283464487139278860/1311014535150436383) used.

## CARLA client

Prerequisites:

- Need to install CARLA Python module

```bash
pip install carla==0.9.15
```

Python verion higher then 3.7 can be used. I used Python 3.10.15, environment requirements file is also included.

Running the client:

```bash
cd carla_sim/CARLA_GymDrive
python carla_simulator_eval.py
```

## CARLA server

CARLA server ran on Azure `Standard NC4as T4 v3 (4 vcpus, 28 GiB memory)`

Source image: `ubuntu-24_04-lts`
Disk: `Standard HDD 64GiB`
IP: `4.210.242.233`

NVIDIA driver setup:

```bash
sudo apt update
sudo apt-get install build-essential

# checking if everything is installed
gcc --version
uname -r

sudo apt update && sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install -y ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-5

sudo reboot
```

CARLA simulator server installation:

```bash
# Python 3.7 needed

# if you use pyenv
pyenv install 3.7
pyenv virtualenv 3.7 carla
pyenv activate carla

sudo apt-get -y install libomp5

wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

mkdir /opt/carla-simulator/
tar -xzvf CARLA_0.9.15.tar.gz -/opt/carla-simulator/C /opt/carla-simulator/

pip install -r /opt/carla-simulator/PythonAPI/examples/requirements.txt
```

Running CARLA sim server:

```bash
cd /opt/carla-simulator
./CarlaUE4.sh -prefernvidia -RenderOffScreen -quality-level=Low -carla-server -benchmark
```

# Sources

- [Carla sim install](https://github.com/carla-simulator/carla/issues/7017#issuecomment-1908462106)
- [Gym wrapper](https://discord.com/channels/1283464486627315815/1283464487139278860/1311014535150436383)
- [NVIDIA driver install](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup#ubuntu)

Other wrappers or stable-baseline3 projects with CARLA:

- [carla-gym](https://github.com/johnMinelli/carla-gym)
- [carla-gym-wrapper](https://github.com/janwithb/carla-gym-wrapper/tree/main)
- [E2E-CARLA-ReinforcementLearning-PPO](https://github.com/gustavomoers/E2E-CARLA-ReinforcementLearning-PPO/tree/main)
- [CARLA-SB3-RL-Training-Environment](https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment)