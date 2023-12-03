import os
import sys
import argparse
import random
import colorsys
import requests
import cv2

from vision_transformer import *
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


patch_size = 8

dino_feat_extractor = vit_small(patch_size=patch_size).cuda()
dino_feat_extractor.load_state_dict(torch.load("dino_deitsmall8_pretrain.pth"))
dino_feat_extractor.eval()
for param in dino_feat_extractor.parameters():
    param.requires_grad = False

img_dir = "/home/prin/gaussian-splatting/output/truck_30k_feat/test/ours_30000/renders"
img_name = "00004.png"
img_path = os.path.join(img_dir, img_name)
img = Image.open(img_path)
img = img.convert('RGB')

# transform = pth_transforms.Compose([
#     pth_transforms.Resize((480, 480)),
#     pth_transforms.ToTensor(),
#     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
# img = transform(img)

img = pil_to_tensor(img).float()
img = to_pil_image(img)
transform = pth_transforms.Compose([
    pth_transforms.Resize((480, 480)),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
img = transform(img)
# img = torch.reshape(img, (3, img.shape[0], img.shape[1])).float()

w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
img = img[:, :w, :h].unsqueeze(0)

w_featmap = img.shape[-2] // patch_size
h_featmap = img.shape[-1] // patch_size

print(img.shape)
attentions = dino_feat_extractor.get_last_selfattention(img.cuda())
nh = attentions.shape[1]

attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
attentions = attentions.reshape(nh, w_featmap, h_featmap)
attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

output_dir = "./output/dino_features"
os.makedirs(output_dir, exist_ok=True)
torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(output_dir, img_name + "_dino.png"))
print(attentions)
for j in range(nh):
    fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
    plt.imsave(fname=fname, arr=attentions[j], format='png')
    print(f"{fname} saved.")