import torch.nn as nn
import torch 
from collections import OrderedDict
import numpy as np
import os

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