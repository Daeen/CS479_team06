import torch.nn as nn
import torch 
from collections import OrderedDict
import numpy as np
import os
from typing import List
import numpy as np
from random import randint
from scene.cameras import Camera
from scene import Scene, GaussianModel
from scipy.spatial.transform import Rotation

class MaskGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=1, num_layers=3):
        super(MaskGenerator, self).__init__()
        
        layers = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())  # Ensure mask values are between 0 and 1
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvModule(nn.Module):
    """Basic convolution module with conv + norm(optional) + activation(optional).

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      ksize(int): size of the convolution kernel (square).
      stride(int): downsampling factor
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(Union[str, None]): nonlinear activation function between convolutions.
      norm_layer(Union[str, None]): normalization to apply between the convolution modules.
    """

    def __init__(self, n_in, n_out, ksize=3, stride=1, pad=True,
                 activation=None, norm_layer=None, padding_mode="reflect", use_bias = False):
        super(ConvModule, self).__init__()

        assert isinstance(
            n_in, int) and n_in > 0, "Input channels should be a positive integer got {}".format(n_in)
        assert isinstance(
            n_out, int) and n_out > 0, "Output channels should be a positive integer got {}".format(n_out)
        assert isinstance(
            ksize, int) and ksize > 0, "Kernel size should be a positive integer got {}".format(ksize)

        layers = OrderedDict()

        padding = (ksize - 1) // 2 if pad else 0
        if padding_mode=="reflect":
            layers["pad"] = nn.ReflectionPad2d(padding)
            padding=0

        layers["conv"] =  nn.Conv2d(n_in, n_out, ksize, stride=stride,
                                          padding=padding, bias=use_bias, padding_mode=padding_mode)

        if norm_layer is not None:
            layers["norm"] = _get_norm_layer(norm_layer, n_out)

        if activation is not None:
            layers["activation"] =  _get_activation(activation)

        # Initialize parameters
        _init_fc_or_conv(layers["conv"], activation)

        self.net = nn.Sequential(layers)

    def forward(self, x):
        x=self.net(x)
        return x
    
class FixupBasicBlock(nn.Module):
    # https://openreview.net/pdf?id=H1gsz30cKX
    expansion = 1

    def __init__(self, n_features, ksize=3, padding=True, padding_mode="reflect",
                 activation="relu", dropout=0.0):
        super(FixupBasicBlock, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = ConvModule(n_features, n_features, ksize=ksize, stride=1,
                                pad=padding, activation=None, norm_layer=None,
                                padding_mode=padding_mode)
        self.dropout1 = nn.Dropout(dropout)

        self.bias1b = nn.Parameter(torch.zeros(1))
        self.activation = _get_activation(activation)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = ConvModule(n_features, n_features, ksize=ksize, stride=1,
                                pad=padding, activation=None, norm_layer=None,
                                padding_mode=padding_mode)
        self.dropout2 = nn.Dropout(dropout)

        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.activation2 = _get_activation(activation)

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.dropout1(out)
        out = self.activation(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = self.dropout2(out)
        out = out * self.scale + self.bias2b

        out += identity
        out = self.activation2(out)

        return out


class FixupResidualChain(nn.Module):
    """Linear chain of residual blocks.
    Args:
      n_features(int): number of input channels.
      ksize(int): size of the convolution kernel (square).
      depth(int): number of residual blocks
      convs_per_block(int): number of convolution per residual block
      activation(str): nonlinear activation function between convolutions.
    """

    def __init__(self, n_features, depth=3, ksize=3, activation="relu", padding_mode="reflect", dropout=0.0):
        super(FixupResidualChain, self).__init__()

        assert isinstance(
            n_features, int) and n_features > 0, "Number of feature channels should be a positive integer"
        assert (isinstance(ksize, int) and ksize > 0) or isinstance(
            ksize, list), "Kernel size should be a positive integer or a list of integers"
        assert isinstance(
            depth, int) and depth > 0 and depth < 16, "Depth should be a positive integer lower than 16"

        self.depth = depth

        # Core processing layers
        layers = OrderedDict()
        for lvl in range(depth):
            blockname="resblock{}".format(lvl)
            layers[blockname]=FixupBasicBlock(
                n_features, ksize=ksize, activation=activation,
                padding_mode=padding_mode, dropout=dropout)

        self.net=nn.Sequential(layers)

        self._reset_weights()

    def _reset_weights(self):
        for m in self.net.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.net.conv.weight, mean=0, std=np.sqrt(2 /
                                (m.conv1.net.conv.weight.shape[0] * np.prod(m.conv1.net.conv.weight.shape[2:]))) * self.depth ** (-0.5))
                nn.init.constant_(m.conv2.net.conv.weight, 0)

    def forward(self, x):
        x = self.net(x)
        return x
    
class FixUpResNet_withMask(nn.Module):
    def __init__(self, in_channels, out_channels, internal_depth, blocks, kernel_size, dropout=0.0, last_activation=None):
        super(FixUpResNet_withMask, self).__init__()
        self.mask_generator_d = MaskGenerator(in_channels // 2 ) 
        self.mask_generator_s = MaskGenerator(in_channels // 2) 

        self.encoder_d = nn.Sequential(
                           ConvModule(in_channels//2, internal_depth//2, ksize=kernel_size, pad=True, activation="relu", norm_layer=None, padding_mode="reflect"),
                           nn.Dropout(dropout),
                           FixupResidualChain(internal_depth//2, ksize=kernel_size, depth=int(blocks/4), padding_mode="reflect", dropout=dropout),
                       )
        self.encoder_s = nn.Sequential(
                           ConvModule(in_channels//2, internal_depth//2, ksize=kernel_size, pad=True, activation="relu", norm_layer=None, padding_mode="reflect"),
                           nn.Dropout(dropout),
                           FixupResidualChain(internal_depth//2, ksize=kernel_size, depth=int(blocks/4), padding_mode="reflect", dropout=dropout),
                       )

        self.decoder = nn.Sequential(
                           FixupResidualChain(internal_depth, ksize=kernel_size, depth=int(blocks/4), padding_mode="reflect", dropout=dropout),
                           ConvModule(internal_depth, out_channels, ksize=kernel_size, pad=True, activation=last_activation, norm_layer=None, padding_mode="reflect")
                       )


    def forward(self, diffuse, specular):

        mask_d = self.mask_generator_d(diffuse)
        mask_s = self.mask_generator_s(specular)

        masked_diffuse = diffuse * mask_d
        masked_specular = specular * mask_s
        diffuse_encoded = self.encoder_d(masked_diffuse)
        specular_encoded = self.encoder_s(masked_specular)
        x_out = self.decoder(torch.cat((diffuse_encoded, specular_encoded), dim=1))
        return x_out

    def plot_histogram(self, tb_writer, path, step):
        for name, weights in self.named_parameters():
            tb_writer.add_histogram(os.path.join(path, name), weights, step)



# Helpers ---------------------------------------------------------------------

def _get_norm_layer(norm_layer, channels):
    valid = ["instance", "batch"]
    assert norm_layer in valid, "norm_layer should be one of {}".format(valid)

    if norm_layer == "instance":
        layer = nn.InstanceNorm2d(channels, affine=True)
    elif norm_layer == "batch":
        layer = nn.BatchNorm2d(channels, affine=True)
    nn.init.constant_(layer.bias, 0.0)
    nn.init.constant_(layer.weight, 1.0)
    return layer


def _get_activation(activation):
    valid = ["relu", "leaky_relu", "lrelu", "elu", "selu", "sigmoid"]
    assert activation in valid, "activation should be one of {}".format(valid)
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu" or activation == "lrelu":
        return nn.LeakyReLU(inplace=True)
    if activation == "elu":
        return nn.ELU(inplace=True)
    if activation == "selu":
        return nn.SELU(inplace=True)
    if activation == "sigmoid":
        return nn.Sigmoid()

    return None


def _init_fc_or_conv(fc_conv, activation):
    gain = 1.0
    if activation is not None:
        try:
            gain = nn.init.calculate_gain(activation)
        except:
            print("Warning using gain of ",gain," for activation: ",activation)
    nn.init.xavier_uniform_(fc_conv.weight, gain)
    if fc_conv.bias is not None:
        nn.init.constant_(fc_conv.bias, 0.0)


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