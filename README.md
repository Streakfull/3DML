# 3DML

This repo contains all related work for the Machine Learning for 3D Geometry Course SS23.

# Project

3D shape reconstruction from 2D images

## Setup

##### This mentions the steps neededed to run the project on the 3DML cluster provided by TUM

- `ssh <user>@ml3d.vc.in.tum.de` - ssh into the login node
  - You will be prompted to enter your password
- `salloc --gpus=1`
- `mkdir /cluster/54/<user>`
- `cd /cluster/54/<user>`
- `git clone https://<github_token_here>@github.com/Streakfull/3DML.git`
- `cd 3DML/Project`
- `poetry install`
- `poetry shell`
- `pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
- `poetry run jupyter notebook --no-browser --ip=0.0.0.0 --port=8888`
- From your local machine: `ssh -NL 3000:TUINI15-<connected_node>.vc.in.tum.de:8888 <user>@ml3d.vc.in.tum.de`
- Run the `initial_setup.ipynb` notebook to obtain the dataset

# General Links

##### This section contains any useful links found

- [3D-R2N2 Repo](https://github.com/chrischoy/3D-R2N2)
- [How to get a github token](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
- [Connecting to 3D ML Cluster](https://www.moodle.tum.de/pluginfile.php/4556578/mod_resource/content/1/ML3D%20Compute%20Cluster.pdf)

# Team

- Biray Sutcuoglu - biray.suetcueoglu@tum.de
- Michael Kubitza - michael.kubitza@tum.de
- Youssef Youssef - youssef@youssef.tum.de
