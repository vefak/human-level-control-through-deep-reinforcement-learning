
## Installation

First, you need build your conda env

```
conda create --name atari python==3.8
conda activate atari
```

Then, the necessary dependencies must be installed. You can see all installations in condalist.txt files

Pytorch installation. It can be different for your device according to CUDA version. Please check Pytorch Installation Page:
https://pytorch.org/get-started/previous-versions/
```
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
```

Gym installation
```
pip install gym[atari]
pip install gym[classic-control]
pip install autorom[accept-rom-license] // Bugfix for old gym enviroments

```
Using msgpack for saving weights
```
pip install msgpack==1.0.2
```
Install tensorboard to see graphs.
```
pip install tensorboard==1.15.0
```

## Train and Run

For training run following code
```
python dqn.py // Start training process
python observe.py // Open gym env and algorithm plays the game
```




