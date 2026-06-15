# T2MAC
Official repository of T2MAC: Targeted and Trusted Multi-Agent Communication Through Selective Engagement and Evidence-Driven Integration


```bash
python src/main.py --config=tmac_p2p_comm --env-config=sc2 with env_args.map_name=6h_vs_8z
```

# TO RUN ON GOOGLE COLAB FOLLOW THIS STEPS:


```bash

import os

# 1. Clean and re-clone the repository
!rm -rf /content/T2MAC
!git clone -q https://github.com/ZangZehua/T2MAC.git
%cd /content/T2MAC

# 2. Explicitly install/verify core libraries + GPU-supported PyTorch
print("Verifying core libraries (torch, numpy, pyyaml)...")
!pip install -q numpy torch torchvision torchaudio pyyaml

# 3. Install required project experiment and logging tools
print("Installing Sacred and Tensorboard dependencies...")
!pip install -q sacred pymongo tensorboard_logger
!pip install -q git+https://github.com/oxwhirl/smac.git

# 4. Apply our custom compatibility patches for modern Python/Colab
!sed -i 's/yaml.load(f)/yaml.safe_load(f)/g' /content/T2MAC/src/main.py
!sed -i 's/collections.Mapping/collections.abc.Mapping/g' /content/T2MAC/src/main.py
!sed -i '1s/^/import datetime\n/' /content/T2MAC/src/modules/agents/tmac_p2p_comm_rnn_msg_agent.py

# 5. Download and extract StarCraft II Linux binary + maps
!mkdir -p /content/T2MAC/3rdparty
%cd /content/T2MAC/3rdparty
print("Downloading and configuring StarCraft II (takes ~1 minute)...")
!wget -q http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
!unzip -q -P iagreetotheeula SC2.4.10.zip -d /content/T2MAC/3rdparty/
!git clone -q https://github.com/oxwhirl/smac.git smac_maps
!mkdir -p /content/T2MAC/3rdparty/StarCraftII/Maps
!cp -r /content/T2MAC/3rdparty/smac_maps/smac/env/starcraft2/maps/SMAC_Maps /content/T2MAC/3rdparty/StarCraftII/Maps/

# 6. Configure environment variables for PySC2
os.environ['SC2PATH'] = '/content/T2MAC/3rdparty/StarCraftII'

# 7. Move back to root and execute on GPU
%cd /content/T2MAC
print("Launching T2MAC on the T4 GPU...")
!python src/main.py --config=tmac_p2p_comm --env-config=sc2 with env_args.map_name=6h_vs_8z use_cuda=True

```