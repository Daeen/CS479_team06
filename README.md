# This is our project on 3D Gaussian Splatting for Real-Time Radiance Field Rendering
Bernhard Kerbl*, Georgios Kopanas*, Thomas Leimkühler, George Drettakis (* indicates equal contribution)<br>
The link to the original project details is given below. The original repository, with given instructions to set up 3D Gaussian splatting, can be accessed from the github link in their website.
| [Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) |<br>

This repository contains the modification on the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

# DINO Code and Pretrained Model
This repository uses a ViT-S/8 DINO pretrained model. Model code (including vision_transformer.py and visualize_dino.py that we slightly modified) can be found [here](https://github.com/facebookresearch/dino) and the ViT-S/8 pretrained DINO weights can be downloaded from this [link](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth).

# Data Sets
The datasets used for experiments are the following. 
- Mip-NeRF 360 Scenes taken from the [link](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) provided by the original Mip-NeRF 360 [webpage](https://jonbarron.info/mipnerf360/)
- Tanks&Temples Truck and Train Scenes and Deep Blending Playroom and Dr. Johnson Scenes downloaded from this [link](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) provided by the original 3D Gaussian splatting [github repo](https://github.com/graphdeco-inria/gaussian-splatting)
- Synthetic NeRF Scenes taken from this [google drive folder](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) provided by the [original NeRF github repo](https://github.com/bmild/nerf)
