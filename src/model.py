"""
2024. 03. 20.

Define the necessary classes for the StyleGAN2 model.

This code was created with reference to the following repositories:
    ・https://github.com/NVlabs/stylegan2/tree/master
    ・https://github.com/ayukat1016/gan_sample.git
    ・https://github.com/rosinality/id-gan-pytorch/tree/master/stylegan2
"""
from typing import Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional


def noise_normalization(input_noise: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize the input noise vector.
    Apply to the input of the Mapping network.

    Parameters
    ----------
    :param input_noise:     Input noise vector.
    :param eps:             Small value to avoid division by zero.
    :return:                Normalized noise vector.
    """
    noise_var = torch.mean(input_noise**2, dim=1, keepdim=True)

    return input_noise / torch.sqrt(noise_var + eps)


class Amplify(nn.Module):
    """
    Amplify the signal of feature map.
    """

    def __init__(self, rate: float) -> None:
        """
        Parameters
        ----------
        :param rate:    Amplification rate.
        """
        super(Amplify, self).__init__()
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.rate


class EqualizedLinear(nn.Module):
    """
    Dense layer with equalized learning rate and custom learning rate multiplier.
    """

    def __init__(self, in_dim: int, out_dim: int, lr_mul: float) -> None:
        """
        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        lr_mul : float
            Learning rate multiplier.
        """
        super(EqualizedLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn((out_dim, in_dim)))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0 / lr_mul)
        self.weight_scale = 1.0 / (in_dim**0.5) * lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = functional.linear(x, weight=self.weight * self.weight_scale, bias=None)
        return out


class AddBiasChannelWise(nn.Module):
    """
    Add bias to the input tensor.
    """

    def __init__(self, out_c: int, bias_scale: float) -> None:
        """
        Parameters
        ----------
        :param out_c:         Output dimension.
        :param bias_scale:    Scale factor of the bias.
        """
        super(AddBiasChannelWise, self).__init__()

        self.bias = nn.Parameter(torch.zeros(out_c))
        torch.nn.init.zeros_(self.bias.data)
        self.bias_scale = bias_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_len, *_ = self.bias.shape
        new_shape = (1, bias_len) if x.ndim == 2 else (1, bias_len, 1, 1)

        y = x + self.bias.view(*new_shape) * self.bias_scale

        return y


class ModulateConv2d(nn.Module):
    """
    Modulated convolution layer.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        w_dim: int,
        kh: int,
        kw: int,
        pad: int,
        stride: int,
        lr_mul: float = 1.0,
        demodulate: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        :param in_c:          The number of input channels.
        :param out_c:         The number of output channels.
        :param kh:            The height of the kernel. 1 in case of ground motion data.
        :param kw:            The width of the kernel.
        :param pad:           The number of padding.
        :param stride:        The width of the stride.
        :param w_dim:         The dimension of the intermediate latent vector.
        :param lr_mul:        Learning rate multiplier.
        :param demodulate:    Whether to demodulate.
        """
        super(ModulateConv2d, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.kh = kh
        self.kw = kw

        self.stride = stride
        self.pad = pad
        self.demodulate = demodulate

        # Initialize the convolution weights.
        self.weight = nn.Parameter(torch.randn(out_c, in_c, kh, kw))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0 / lr_mul)
        self.weight_scale = 1 / np.sqrt(in_c * kh * kw) * lr_mul

        # Layer for affine transformation of style
        self.linear = EqualizedLinear(in_dim=w_dim, out_dim=in_c, lr_mul=lr_mul)
        self.bias = AddBiasChannelWise(out_c=in_c, bias_scale=lr_mul)

    def forward(self, pack: List[torch.Tensor]) -> torch.Tensor:
        x, style = pack
        b_size, c_size, hei, wid = x.shape

        # Transform the style through an affine transformation to create data for modulation.
        # [b, 512] -> [b, 512]
        mod_style = self.linear(style)
        mod_style = self.bias(mod_style) + 1

        # Change the shape of the mod_style to make it applicable to the weights of the convolution.
        # [b, in_c] -> [b, 1, in_c, 1, 1]
        mod_style = mod_style.view(b_size, 1, self.in_c, 1, 1)

        # Match the shape of the convolution weights to that of mod_style.
        # [out_c, in_c, kh, kw] -> [1, out_c, in_c, kh, kw]
        resized_weight = self.weight.view(1, self.out_c, self.in_c, self.kh, self.kw)

        # Apply the style to the weight.
        # weight          : 1,     out_c, in_c, kh, kw
        # style           : batch,     1, in_c,  1,  1
        # modulated_weight: batch, out_c, in_c, kh, kw
        modulated_weight = resized_weight * mod_style * self.weight_scale

        # If normalizing again, demodulate=True.
        # If not normalizing, use the modulated_weight as it is.
        if self.demodulate:
            # Normalize for each channel, so take the sum of squares with respect to in_c, kh, kw.
            weight_sum = modulated_weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8
            demodulate_norm = torch.rsqrt(weight_sum).view(b_size, self.out_c, 1, 1, 1)
            weight = modulated_weight * demodulate_norm
        else:
            weight = modulated_weight

        # Convolution
        # Input shape : [1, b*channel, H, W]
        # Weight shape: [b*out_c, in_c, kh, kw]
        weight = weight.view(b_size * self.out_c, self.in_c, self.kh, self.kw)
        x = x.view(1, b_size * c_size, hei, wid)

        out = functional.conv2d(x, weight=weight, padding=(0, self.pad), stride=self.stride, groups=b_size)
        out = out.view(b_size, self.out_c, hei, wid)

        return out


class UpSampleConv2d(nn.Module):
    """
    Up-sample the feature map with modulated convolution.
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kh: int,
        kw: int,
        pad: int,
        stride: int,
        w_dim: int,
        lr_mul: float = 1.0,
        demodulate: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        :param in_c:          The number of input channels.
        :param out_c:         The number of output channels.
        :param kh:            The height of the kernel. 1 in case of ground motion data.
        :param kw:            The width of the kernel.
        :param pad:           The number of padding.
        :param stride:        The width of the stride.
        :param w_dim:         The dimension of the intermediate latent vector.
        :param lr_mul:        Learning rate multiplier.
        :param demodulate:    Whether to demodulate.
        """
        super(UpSampleConv2d, self).__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.kh = kh
        self.kw = kw

        self.stride = stride
        self.pad = pad
        self.demodulate = demodulate

        # Initialize the convolution weights.
        # When using "conv_transpose2d", be aware that the positions of the input and output channels for the weights
        # are reversed compared to "conv2d".
        self.weight = nn.Parameter(torch.randn(in_c, out_c, kh, kw))
        torch.nn.init.normal_(self.weight.data, mean=1.0, std=1.0 / lr_mul)
        self.weight_scale = 1.0 / np.sqrt(in_c * kh * kw) * lr_mul

        # Layer for affine transformation of style
        self.linear = EqualizedLinear(in_dim=w_dim, out_dim=in_c, lr_mul=lr_mul)
        self.bias = AddBiasChannelWise(out_c=in_c, bias_scale=lr_mul)

    def forward(self, pack: List[torch.Tensor]) -> torch.Tensor:
        x, style = pack
        b_size, c_size, hei, wid = x.shape

        # Transform the style through an affine transformation to create data for modulation.
        # [b, 512] -> [b, 512]
        mod_style = self.linear(style)
        mod_style = self.bias(mod_style) + 1

        # Apply the style to the weight. Same as ModulateConv2d.
        # weight          : 1,      in_c, out_c, kh, kw
        # style           : batch,  in_c,     1,  1,  1
        # modulated_weight: batch,  in_c, out_c, kh, kw
        mod_style = mod_style.view(b_size, self.in_c, 1, 1, 1)
        resized_weight = self.weight.view(1, self.in_c, self.out_c, self.kh, self.kw)
        modulated_weight = resized_weight * mod_style * self.weight_scale

        # If normalizing again, demodulate=True.
        # If not normalizing, use the modulated_weight as it is.
        # Same as ModulateConv2d.
        if self.demodulate:
            weight_sum = modulated_weight.pow(2).sum(dim=[1, 3, 4]) + 1e-8
            demodulate_norm = torch.rsqrt(weight_sum).view(b_size, 1, self.out_c, 1, 1)
            weight = modulated_weight * demodulate_norm
        else:
            weight = modulated_weight

        # Convolution
        # Input shape : [1, b*channel, H, W]
        # Weight shape: [b*in_c, out_c, kh, kw]
        weight = weight.view(b_size * self.in_c, self.out_c, self.kh, self.kw)
        x = x.view(1, b_size * c_size, hei, wid)

        out = functional.conv_transpose2d(x, weight=weight, padding=self.pad, stride=self.stride, groups=b_size)

        _, _, temp_h, temp_w = out.shape
        out = out.view(b_size, self.out_c, temp_h, temp_w)

        return out


class BlurPooling(nn.Module):
    def __init__(self, in_c: int) -> None:
        """
        Parameters
        ----------
        :param in_c: The number of channels.
        """
        super(BlurPooling, self).__init__()

        # Create a filter for blur pooling.
        # The filter is [1, 3, 3, 1] for 1 channel.
        blur_kernel = np.array([1.0, 3.0, 3.0, 1.0])
        blur_filter = torch.Tensor(blur_kernel)
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter.expand(in_c, 1, 1, 4)
        self.register_buffer("const_blur_filter", blur_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c_size, _, _ = x.shape
        # Pad with zeros on both sides.
        x = functional.pad(x, (1, 1), mode="constant", value=0)
        # convolution
        out = functional.conv2d(x, weight=self.const_blur_filter, stride=1, padding=0, groups=c_size)

        return out


class UpSampleWave(nn.Module):
    """
    Up-sample the wave data with blur pooling.

    !! Attention !!
    The up-sampling factor is fixed at 2.
    """

    def __init__(self, in_c: int = 1) -> None:
        """
        Parameters
        ----------
        :param in_c: The number of channels of generated ground motion data.
                     If the target ground motion data is single-component, in_c=1.
                     If it is three-component, in_c=3.
                     However, note that this code has only been debugged for the case in_c=1.
        """
        super(UpSampleWave, self).__init__()

        # Create a filter for blur pooling. Same as BlurPooling.
        blur_kernel = np.array([1.0, 3.0, 3.0, 1.0])
        blur_filter = torch.Tensor(blur_kernel)
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter.expand(in_c, 1, 1, 4)
        self.register_buffer("const_blur_filter", blur_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_size, c_size, hei, wid = x.shape

        # Change the data shape.
        # [b, c, 1, n] -> [b, c, n, 1]
        x = x.reshape(b_size, c_size, wid, hei)

        # Pad zeros to increase the number of data.
        x = functional.pad(x, (0, 1), mode="constant", value=0)

        # Reshape
        # [b, 1, n, 2] -> [b, 1, 1, n*2]
        x = x.reshape(b_size, c_size, hei, -1)

        # Pad two zeros on the left and one zero on the right.
        x = functional.pad(x, (2, 1), mode="constant", value=0)

        # Convolution
        out = functional.conv2d(x, weight=self.const_blur_filter, stride=1, padding=0, groups=c_size)

        return out


class PixelWiseNoise(nn.Module):
    """
    Add noise in the temporal direction of the generated wave data.
    The same noise is added to all channels.
    Base noise is initialized with Gaussian noise and is not updated during training.
    Scale factor of the noise is updated.
    """

    def __init__(self, wave_width: int) -> None:
        """
        Parameters
        ----------
        :param wave_width: The width (temporal direction) of the generated wave data.
        """
        super(PixelWiseNoise, self).__init__()
        self.register_buffer("const_noise", torch.randn(1, 1, 1, wave_width))
        self.noise_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_size, _, hei, wid = x.shape
        noise = self.const_noise.expand(b_size, 1, hei, wid)

        out = x + noise * self.noise_scale

        return out


# ---------------- Class for Discriminator ----------------------------------------------------------------------------
class Conv2dLayer(nn.Module):
    """
    Convolution layer with equalized learning rate and custom learning rate multiplier.
    """

    def __init__(self, in_c: int, out_c: int, kh: int, kw: int, pad: int, stride: int, lr_mul: float) -> None:
        """
        Parameters
        ----------
        :param in_c:      The number of input channels.
        :param out_c:     The number of output channels.
        :param kh:        Kernel height.
        :param kw:        Kernel width.
        :param pad:       Padding width of the convolution.
        :param stride:    Stride width of the convolution.
        :param lr_mul:    Learning rate multiplier.
        """
        super(Conv2dLayer, self).__init__()

        self.stride = stride
        self.pad = pad

        # Initialize the weight and scaling factor.
        self.weight = nn.Parameter(torch.randn(out_c, in_c, kh, kw))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0 / lr_mul)
        self.weight_scale = 1 / np.sqrt(in_c * kh * kw) * lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = functional.conv2d(x, weight=self.weight * self.weight_scale, padding=(0, self.pad), stride=self.stride)
        return out


class FromWave(nn.Module):
    """
    Convert wave data to feature map.
    """

    def __init__(self, in_c: int, out_c: int, lr_mul: float) -> None:
        """
        Parameters
        ----------
        :param in_c:   The number of input channels.
        :param out_c:  The number of output channels.
        :param lr_mul: Learning rate multiplier.
        """
        super(FromWave, self).__init__()
        self.conv2d = Conv2dLayer(
            in_c=in_c,
            out_c=out_c,
            kh=1,
            kw=1,
            pad=0,
            stride=1,
            lr_mul=1.0,
        )
        self.bias = AddBiasChannelWise(out_c=out_c, bias_scale=lr_mul)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(x)
        out = out + self.bias(out)
        out = self.activation(out)
        return out


class DownSampleConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, kw: int) -> None:
        """
        Parameters
        ----------
        :param in_c:   The number of input channels.
        :param out_c:  The number of output channels.
        :param kw:     Kernel width.
        """
        super(DownSampleConv2d, self).__init__()

        # Initialize the weight and scaling factor.
        self.weight = nn.Parameter(torch.randn(out_c, in_c, 1, kw))
        torch.nn.init.normal_(self.weight.data, mean=0.0, std=1.0)
        self.weight_scale = 1 / np.sqrt(in_c * 1 * kw) * 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = functional.conv2d(input=x, weight=self.weight * self.weight_scale, padding=(0, 0), stride=(1, 2))

        return out


class BlurPoolDiscriminator(nn.Module):
    def __init__(self, in_c: int, pad: int) -> None:
        """
        Parameters
        :param in_c:  The number of input channels.
        :param pad:   Padding number.
        """
        super(BlurPoolDiscriminator, self).__init__()

        blur_kernel = np.array([1.0, 3.0, 3.0, 1.0])
        blur_filter = torch.Tensor(blur_kernel)
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter.expand(in_c, 1, 1, 4)
        self.register_buffer("const_blur_filter", blur_filter)

        self.pad_num = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c_size, _, _ = x.shape

        # Add two zeros at both ends.
        x = functional.pad(x, (self.pad_num, self.pad_num), mode="constant", value=0)
        # print(x.shape)

        out = functional.conv2d(x, weight=self.const_blur_filter, stride=1, padding=0, groups=c_size)

        return out


class ResBlockDiscriminator(nn.Module):
    """
    Residual block for discriminator.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        """
        Parameters
        ----------
        :param in_c:   The number of input channels.
        :param out_c:  The number of output channels.
        """
        super(ResBlockDiscriminator, self).__init__()

        self.main_model = nn.Sequential(
            Conv2dLayer(in_c=in_c, out_c=in_c, kh=1, kw=3, pad=1, stride=1, lr_mul=1.0),
            AddBiasChannelWise(out_c=in_c, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
            BlurPoolDiscriminator(in_c=in_c, pad=2),
            DownSampleConv2d(in_c=in_c, out_c=out_c, kw=3),
            AddBiasChannelWise(out_c=out_c, bias_scale=1.0),
            Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.blur_pool_skip = BlurPoolDiscriminator(in_c=in_c, pad=1)
        self.down_sample_conv_skip = DownSampleConv2d(in_c=in_c, out_c=out_c, kw=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.main_model(x)
        skip = self.blur_pool_skip(skip)
        skip = self.down_sample_conv_skip(skip)
        # print(x.shape)
        # print(skip.shape)

        out = (x + skip) * (1.0 / np.sqrt(2))

        return out


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int, num_features: int) -> None:
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, 512, 1, 4]
        my_group_size = min(self.group_size, x.shape[0])
        _, channel_d, height_d, width_d = x.shape

        # y: [group_size, b/group_size, num_features, 512/num_features, 1, 4]
        y = x.view((my_group_size, -1, self.num_features, channel_d // self.num_features, height_d, width_d))

        # Subtract the mean
        # y.mean: [1, 8, 1, 512, 1, 4]
        y = y - y.mean(dim=0, keepdim=True)

        # Std. for each group
        y = y * y
        y = y.mean(dim=0)
        y = torch.sqrt(y + 1e-8)

        # Calculate the mean for data
        y = y.mean(dim=[2, 3, 4], keepdim=True)

        # Reduce the dimension
        y = y.mean(dim=2)

        # change the shape
        y = y.repeat([my_group_size, 1, height_d, width_d])

        out = torch.cat([x, y], dim=1)

        return out


def adjust_adam_params(
    reg_interval: int, learning_rate: float, beta_1: float, beta_2: float
) -> Tuple[float, float, float]:
    """
    Adjust Adam parameters for mini-batch training.
    This function is based on the paper https://arxiv.org/abs/1912.04958, Appendix B, Lazy regularization.
    :param reg_interval:     Frequency of applying regularization error.
    :param learning_rate:    Base learning rate.
    :param beta_1:           Hyperparameter beta_1 of Adam.
    :param beta_2:           Hyperparameter beta_2 of Adam.
    :return:                 Adjusted learning rate, beta_1, beta_2.
    """
    mini_batch_ratio = reg_interval / (reg_interval + 1)
    l_rate = learning_rate * mini_batch_ratio
    b1 = beta_1**mini_batch_ratio
    b2 = beta_2**mini_batch_ratio
    return l_rate, b1, b2


# function for training
def set_model_requires_grad(model: nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def generator_logistic_loss(disc_fake_out: torch.Tensor) -> torch.Tensor:
    """
    Calculate the logistic loss for the generator.
    :param disc_fake_out: Output of the discriminator inputting the generated data.
    :return:              Logistic loss for the generator.
    """
    out = torch.mean(functional.softplus(-disc_fake_out))

    return out


def discriminator_logistic_loss(disc_fake_out: torch.Tensor, disc_real_out: torch.Tensor) -> torch.Tensor:
    """
    Calculate the logistic loss for the discriminator.
    :param disc_fake_out: Output of the discriminator inputting the generated data.
    :param disc_real_out: Output of the discriminator inputting the real data.
    :return:              Logistic loss for the discriminator.
    """
    out = torch.mean(functional.softplus(disc_fake_out)) + torch.mean(functional.softplus(-disc_real_out))

    return out


def generator_wgan_loss(disc_fake_out: torch.Tensor) -> torch.Tensor:
    """
    Calculate the WGAN loss for the generator.
    :param disc_fake_out: Output of the discriminator inputting the generated data.
    :return:              WGAN loss for the generator.
    """
    out = -torch.mean(disc_fake_out)

    return out


def discriminator_wgan_loss(disc_fake_out: torch.Tensor, disc_real_out: torch.Tensor) -> torch.Tensor:
    """
    Calculate the WGAN loss for the discriminator.
    :param disc_fake_out: Output of the discriminator inputting the generated data.
    :param disc_real_out: Output of the discriminator inputting the real data.
    :return:              WGAN loss for the discriminator.
    """
    out = torch.mean(disc_fake_out) - torch.mean(disc_real_out)

    return out


# def gradient_penalty(
#     disc: nn.Module,
#     real_data: torch.Tensor,
#     fake_data: torch.Tensor,
#     real_label: torch.Tensor,
#     fake_label: torch.Tensor,
#     wei: float = 10.0,
# ) -> torch.Tensor:
#     """
#     Calculate the gradient penalty.
#     :param disc:       Discriminator.
#     :param real_data:  Real data.
#     :param fake_data:  Generated data.
#     :param real_label: Label for real data.
#     :param fake_label: Label for generated data.
#     :param wei:        Weight of the gradient penalty.
#     :return:           Gradient penalty.
#     """
#     alpha_size = tuple((len(real_data), *(1,) * (real_data.dim() - 1)))
#     alpha_t = torch.Tensor
#     alpha = alpha_t(*alpha_size).to(real_data.device).uniform_()
#
#     x_hat = (real_data.detach() * alpha + fake_data.detach() * (1 - alpha)).requires_grad_()
#
#     def eps_norm(x: torch.Tensor) -> torch.Tensor:
#         x = x.view(len(x), -1)
#         out = (x * x + 1e-15).sum().sqrt()
#         return out
#
#     def bi_penalty(x: torch.Tensor) -> torch.Tensor:
#         return (x - 1) ** 2
#
#     disc_out_x_hat = disc(x_hat)
#     grad_x_hat = torch.autograd.grad(
#         disc_out_x_hat,
#         x_hat,
#         grad_outputs=torch.ones(disc_out_x_hat.size()).to(real_data.device),
#         create_graph=True,
#         only_inputs=True,
#     )[0]
#     penalty = wei * bi_penalty(eps_norm(grad_x_hat)).mean()
#     return penalty


def discriminator_loss_r1(
    disc_real_out: torch.Tensor, reals_f: torch.Tensor, gamma_f: float = 10, d_reg_int: int = 16
) -> torch.Tensor:
    real_grads = torch.autograd.grad(outputs=torch.sum(disc_real_out), inputs=reals_f, create_graph=True)[0]
    penalty = torch.sum(real_grads**2, dim=[1, 2, 3])
    reg = (penalty * gamma_f * 0.5 * d_reg_int).mean()
    return reg


class GeneratorLossPathRegularization(nn.Module):
    def __init__(self, device, pl_decay: float = 0.01, pl_weight: float = 2.0, g_reg_int: int = 4) -> None:
        super(GeneratorLossPathRegularization, self).__init__()

        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean_var = torch.zeros((1,)).to(device)
        self.reg_interval = g_reg_int

        self.device = device

    def forward(self, fake_wave: torch.Tensor, fake_style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pl_noise = torch.randn(fake_wave.shape) / np.sqrt(np.prod(fake_style.shape[2:]))
        pl_noise = pl_noise.to(self.device)
        f_img_out_pl_n = torch.sum(fake_wave * pl_noise)
        pl_grads = torch.autograd.grad(outputs=f_img_out_pl_n, inputs=fake_style, create_graph=True)[0]
        pl_grads_sum_mean = pl_grads.pow(2).sum(dim=2).mean(dim=1)
        pl_length = torch.sqrt(pl_grads_sum_mean)

        # Track exponential moving average of |J*y|.
        pl_mean = self.pl_mean_var + self.pl_decay * (pl_length.mean() - self.pl_mean_var)
        self.pl_mean_var = pl_mean.detach()

        # Calculate (|J*y|-a)^2.
        pl_penalty = (pl_length - pl_mean).pow(2).mean()
        reg = pl_penalty * self.pl_weight * self.reg_interval
        return reg, pl_length.mean()


if __name__ == "__main__":
    # test for ModulateConv2d ----------------------------------------------
    print("----------------------- Test for ModulateConv2d -----------------------")
    test_mod_conv = ModulateConv2d(
        in_c=512,
        out_c=512,
        kh=1,
        kw=3,
        stride=1,
        pad=1,
        demodulate=True,
        lr_mul=1.0,
        w_dim=512,
    )

    print("----------------------- Training parameters of ModulateConv2d -----------------------")
    for name, param in test_mod_conv.named_parameters():
        print(f"{name}: {param.size()}")
    print("--------------------------------------------------------------------------------------")

    test_x = torch.randn(3, 512, 1, 8)
    test_w = torch.randn(3, 512)
    test_out = test_mod_conv([test_x, test_w])
    print(test_out.size())
    print("-----------------------------------------------------------------------")
    print()

    # test for UpSampleConv2d ----------------------------------------------
    print("----------------------- Test for UpSampleConv2d -----------------------")
    test_up_conv = UpSampleConv2d(
        in_c=512,
        out_c=512,
        kh=1,
        kw=3,
        stride=2,
        pad=0,
        demodulate=True,
        lr_mul=1.0,
        w_dim=512,
    )

    print("----------------------- Training parameters of UpSampleConv2d -----------------------")
    for name, param in test_up_conv.named_parameters():
        print(f"{name}: {param.size()}")
    print("--------------------------------------------------------------------------------------")

    test_x = torch.randn(3, 512, 1, 8)
    test_w = torch.randn(3, 512)
    test_out = test_up_conv([test_x, test_w])
    print(test_out.size())
    print("-----------------------------------------------------------------------")
    print()

    # test for UpSampleWave ----------------------------------------------
    print("----------------------- Test for UpSampleWave -----------------------")
    test_up_wave = UpSampleWave(in_c=1)

    test_wave = torch.randn(3, 1, 1, 8)
    test_out = test_up_wave(test_wave)
    print(test_out.size())
