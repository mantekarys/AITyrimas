# AITyrimas

Metadrive setup commands:

```bash
git clone https://github.com/metadriverse/
cd metadrive/
pip install -e .

# install other required packages
pip install stable_baselines3 mlflow rich PyOpenGL

# torch install (for CUDA 12)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
pip install cuda-python

# test if everything works
cd metadrive/examples/
python verify_image_observation.py --cuda
```