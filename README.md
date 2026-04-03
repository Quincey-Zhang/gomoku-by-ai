# engine configuarion
<br>

## hardware

| | |
| - | - |
| platform | Google Cloud |
| GPU | H100 80GB |
| vCPU | 26 |
| hard disk | 512GB |
<br>

## system
<br>

### install tmux

```
sudo apt-get update
sudo apt-get install tmux
```
<br>

### install anaconda
<br>

install:
```
sudo apt-get update
sudo apt-get install wget
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
sha256sum Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh
```

reference:

https://www.hostinger.com/tutorials/how-to-install-anaconda-on-ubuntu/

<br>

solution to 'conda: command not found':
```
vim ~/.bashrc

add at the end:
export PATH=$PATH:/home/{user_name}/anaconda3/bin

source ~/.bashrc
```
reference:

https://huaweicloud.csdn.net/63a57178b878a54545947693.html

<br>

get Jupyter link:
```
jupyter lab --ip 0.0.0.0 --port 8899 --allow-root

get:
http://<machine external IP>:8899/lab?token=<token>
```
<br>

### install NVIDIA driver
<br>

prerequisites:
```
sudo apt-get install gcc
sudo apt-get install make

sudo apt-get install linux-headers-`uname -r`
```
<br>

solution to linux-headers installation failure:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

reboot

sudo apt-get install linux-headers-`uname -r`
```

reference:

https://www.soinside.com/question/fRpLiptQ3uzAfgGFSVrgpd

<br>

install:
```
sudo sh NVIDIA-Linux-x86_64-570.133.07.run
```
<br><br>

# training environment
<br>

## CUDA
<br>

download and install:

https://developer.nvidia.com/cuda-12-4-1-download-archive

<br>

solution to 'add-apt-repository command not found':
```
sudo apt-get install software-properties-common
```
<br>

check:
```
nvcc --version
```
<br>

solution to 'nvcc command not found':
```
cd /usr/local/cuda/bin
vim ~/.bashrc

add at the end:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

source ~/.bashrc
```

reference:

https://blog.csdn.net/Flying_sfeng/article/details/103343813

<br>

## apt-get install
<br>

```
sudo apt-get install git-all
sudo apt-get install cmake
sudo apt-get install libx11-dev
sudo apt-get install libxrandr-dev
sudo apt-get install libxinerama-dev
sudo apt-get install libxcursor-dev
sudo apt-get install libxi-dev
```
<br>

## conda environment
<br>

### Python
```
conda create -n madrona python=3.10
conda activate madrona
```
<br>

### pip install
```
pip install -r requirements.txt
```
<br>

### run training
<br>

get external madrona module:
```
git submodule add https://github.com/shacklettbp/madrona.git external/madrona
```

Make sure the 'CMakeLists.txt' file is in the 'external' folder, which content:
```
add_subdirectory(madrona EXCLUDE_FROM_ALL)
```
<br>

compile C++ simulation environment:
```
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make -j
cd ..
```
<br>

run:
```
cd train

MADRONA_MWGPU_KERNEL_CACHE=/tmp/gomoku_cache python train_league.py --cuda
```
<br>

### run webgame
<br>

```
cd webgame
python server.py
```
<br><br>

# models
<br>

## training records
<br>

| model name | training parameters | training time |
| - | - | - |
| league_288 | num_envs: 2048 <br> num_steps: 128 <br> num_updates: 1e6 <br> learning_rate: 1e-4 <br> anneal_lr: True <br> num_minibatches: 4 <br> ent_coef: 1e-2 <br> selfplay_ratio: 0.2 <br><br> league threshold: 0.9 <br> league interval: 10 <br> save interval: 1e4 | 14h 58min |
<br>

## model VS model
<br>

```
python evaluation/ai_vs_ai.py --model-a train/gomoku_league_output/models/league_288.pkl --model-b train/gomoku_league_output/models/league_288.pkl --num-games 1000
```
