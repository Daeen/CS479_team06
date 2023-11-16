import torch
import torch.nn as nn
from typing import List
import os
import numpy as np
from random import randint
from scene.cameras import Camera
from scene import Scene, GaussianModel
from scipy.spatial.transform import Rotation

class BasicMLP(nn.Module):
    def __init__(self, insize, outsize, width, num_layers, activate_last=False):
        super(BasicMLP, self).__init__()
        self.width = width
        self.num_layers = num_layers

        if num_layers>1:
            first_layer = nn.Linear(insize, width)
            nn.init.kaiming_uniform_(first_layer.weight, nonlinearity="relu")
            layers = [first_layer]  # , nn.ReLU()]
            for i in range(num_layers - 1):
                layer = nn.Linear(width, width)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                layers.append(layer)
            self.lastlayer = nn.Linear(width, outsize, bias=True)
        else:
            layers=[]
            self.lastlayer = nn.Linear(insize, outsize, bias=True)
        nn.init.kaiming_uniform_(self.lastlayer.weight, nonlinearity="relu")

        if activate_last:
            self.lastactivate = nn.ReLU()
        else:
            self.lastactivate = nn.Identity()

        self.module_list = nn.ModuleList(layers)
        self.activate = nn.ReLU()

    def forward(self, x):
        for f in self.module_list:
            x = self.activate(f(x))
        return self.lastactivate(self.lastlayer(x))

class GaussianWarpMLP(nn.Module):
    def __init__(self, insize, outsize, width, num_layers, activate_last=False):
        super(GaussianWarpMLP, self).__init__()
        # Assuming insize accounts for the concatenated input size of camera rotation, translation, and Gaussian position
        self.mlp = BasicMLP(insize, outsize, width, num_layers, activate_last)

    #viewpoint_stack = scene.getTrainCameras().copy()
    #viewpoint = viewpoint_stack[i]
    # base_image = base_viewpoint.original_image
    # base_FoVx, base_FoVy, base_img_w, base_img_h = base_viewpoint.FoVx, base_viewpoint.FoVy, base_viewpoint.image_width, base_viewpoint.image_height
    # base_colmapID, base_trans, base_scale, base_device = base_viewpoint.colmap_id, base_viewpoint.trans, base_viewpoint.scale, base_viewpoint.data_device
    # view_R = Rotation.random(3).as_euler('zxy', degrees=True) / 180.
    # view_T = np.random.rand(3)
    # review_viewpoint = Camera(colmap_id=base_colmapID,R=view_R, T=view_T,FoVx=base_FoVx, FoVy=base_FoVy, image=base_image,gt_alpha_mask=None, image_name=None, uid=0, trans=base_trans, scale=base_scale, data_device=base_device)
    def forward(self, review_viewpoint, pc : GaussianModel):
        camera_rotation = review_viewpoint.R
        camera_translation = review_viewpoint.T
        gaussian_position = pc.get_xyz
        x = torch.cat([camera_rotation, camera_translation, gaussian_position], dim=1)
        return 0.01 * self.mlp(x) + gaussian_position


class TwoBlockMLPGaussian(nn.Module):
    def __init__(self, insize1, insize2, outsize, width, num_layers1, num_layers2):
        super(TwoBlockMLPGaussian, self).__init__()
        self.mlp1 = BasicMLP(insize1, width, width, num_layers1, activate_last=True)
        self.mlp2 = BasicMLP(width + insize2, outsize, width + insize2, num_layers2, activate_last=False)

    def forward(self, review_viewpoint, pc : GaussianModel):
        camera_rotation = review_viewpoint.R
        camera_translation = review_viewpoint.T
        gaussian_position = pc.get_xyz
        b = self.mlp1(gaussian_position)
        return self.mlp2(torch.cat((camera_rotation, camera_translation, b), dim=1))


class CatacausticMLP(nn.Module):
    def __init__(self, width_param, width_embed, depth_param, depth_embed,mean, std, cam_mean, cam_std):
        super(CatacausticMLP, self).__init__()
        self.mlp_param = BasicMLP(3, 2, 16, 3, activate_last=False)
        #this is the warping field
        self.mlp_embed = BasicMLP(5, 3, 256, 4, activate_last=False)
        # self.uv_activate = nn.Sigmoid()
        self.uv_activate = nn.Identity()
        self.uv = None
        self.raw_mlp_out = None
        self.output = None

        self.m = mean
        self.s = std
        self.cam = cam_mean
        self.cas = cam_std

    def normalize_points(self, p):
        return (p - self.m) / self.s
    
    def normalize_cams(self, c):
        return (c - self.cam) / self.cas

    def forward(self, view, pc : GaussianModel):
        points_xyz = self.normalize_points(view.get_xyz)
        camera_center = self.normalize_cams(view.camera_center)
        #input_param = torch.cat((camera_center, points_xyz), dim=1)
        #uv_raw = self.mlp_param(input_param)
        uv_raw = self.mlp_param(camera_center)
        self.uv = self.uv_activate(uv_raw)
        input_embed = torch.cat((self.uv, points_xyz), dim=1)
        #input_embed = self.uv
        self.raw_mlp_out = self.mlp_embed(input_embed)
        self.output = (points_xyz + 0.01 * self.raw_mlp_out)* self.s + self.m
        return self.output


class ProgressiveCatacausticMLP(nn.Module):
    def __init__(self, width_param, width_embed, depth_param, depth_embed,mean, std, cam_mean, cam_std):
        super(ProgressiveCatacausticMLP, self).__init__()
        self.mlp_param = BasicMLP(3, 3, 16, 2, activate_last=False)
        self.mlp_embed = BasicMLP(6, 3, 256, 4, activate_last=False)
        self.uvw_activate = nn.Tanh()
        #self.uv_activate = nn.Identity()
        self.uv = None
        self.uvw = None
        self.raw_mlp_out = None
        self.output = None

        self.m = mean
        self.s = std
        self.cam = cam_mean
        self.cas = cam_std

    def normalize_points(self, p):
        return (p - self.m) / self.s
    
    def normalize_cams(self, c):
        return (c - self.cam) / self.cas

    def sort_uvw(self, uvw):

        def permute_2d(x, permutation):
            d1, d2 = x.size()
            ret = x[
                torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
                permutation.flatten()
            ].view(d1, d2)
            return ret

        uvw_sort_indices = torch.argsort(torch.abs(uvw), dim=1, descending=True)
        return permute_2d(uvw, uvw_sort_indices)

    def forward(self, view, pc : GaussianModel, w_damp_factor=0.1):
        
        # encoder
        # TODO: don't use expanded cameras, we likely just need one
        points_xyz = self.normalize_points(pc.get_xyz)
        camera_center = self.normalize_cams(view.camera_center)
        uvw_raw = self.mlp_param(camera_center)
        uvw = self.uvw_activate(uvw_raw)

        # sort 3D latent code and dampen the smallest value
        #uvw_sorted = self.sort_uvw(uvw)
        #uvw_sorted[:, 2] *= w_damp_factor

        uvw_sorted = uvw
        
        # make sure self.uv is in the pipeline so that we can differentiate w.r.t. it
        self.uv = uvw_sorted[:, :2]
        self.uvw = torch.cat((self.uv, uvw_sorted[:, 2:]), dim=1)

        # decoder
        input_embed = torch.cat((self.uvw, points_xyz), dim=1)
        self.raw_mlp_out = self.mlp_embed(input_embed)
        self.output = (points_xyz + 0.01 * self.raw_mlp_out)* self.s + self.m
        return self.output