ENV setup commands:

```bash
# create Python env (used pyenv)
pyenv install 3.10.15
pyenv virtualenv 3.10.15 <env-name>

# install modules
# Source: https://metadrive-simulator.readthedocs.io/en/latest/install.html#install-metadrive-with-advanced-offscreen-rendering
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install cupy-cuda12x
pip3 install cuda-python
pip3 install PyOpenGL PyOpenGL_accelerate rich tqdm mlflow

# make sure nvidia drivers are working
sudo nvidia-smi

# Run mlflow server
mlflow server --host 127.0.0.1 --port 5000
```