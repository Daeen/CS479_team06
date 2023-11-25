import os
import torch
from torch.optim import Adam, SGD
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from network import FixUpResNet_withMask, WarpFieldMLP

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from network import FixUpResNet_withMask, WarpFieldMLP

def render_set(model_path, name, iteration, views, gaussians, reflection_gaussians, warp_net, combination_net, pipeline, background):
    render_path = os.path.join(model_path, name, "ref/ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ref/ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, None, pipeline, background)["render"]
        reflection_rendering = render(view, reflection_gaussians, warp_net, pipeline, background)["render"]

        diffuse = combination_net(rendering, reflection_rendering, diffuse_ratio=1.0, specular_ratio=0.0)
        reflection = combination_net(rendering, reflection_rendering, diffuse_ratio=0.0, specular_ratio=1.0)


        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(diffuse, os.path.join(render_path, '1{0:04d}'.format(idx) + ".png"))
        torchvision.utils.save_image(reflection, os.path.join(render_path, '2{0:04d}'.format(idx) + ".png"))
        
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        reflection_gaussians = GaussianModel(dataset.sh_degree)
        reflection_scene = Scene(dataset, reflection_gaussians, load_iteration=iteration, shuffle=False)

        warp_net = torch.load(scene.model_path + "/warp_chkpnt" + str(scene.loaded_iter) + ".pth")
        combination_net = torch.load(scene.model_path + "/combination_chkpnt" + str(scene.loaded_iter) + ".pth")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, reflection_gaussians, warp_net, combination_net, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)