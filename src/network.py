"""
2024. 03. 20.

Define the neural network for StyleGAN2, including the following classes:
    ・GroundMotionDatasets: For loading ground motion data and label data.
    ・MappingNetwork: A network that generates intermediate latent variables from latent variables.
    ・LabelPredNetwork: A network that generates corresponding label data in addition to ground motion data.
    ・SynthesisNetwork: A network that generates ground motion data from intermediate latent variables.
    ・Generator: A main network that generates ground motion data and label data.
    ・Discriminator: The discriminator for GANs.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import model


torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


# Dataset
class GroundMotionDatasets(Dataset):
    """
    Dataset class for ground motion data and label data.
    """
    def __init__(self, csv_path: str, label_list: List[str]) -> None:
        """
        Parameters
        ----------
        :param csv_path: str
            The path to the csv file containing the path of ground motion data and the values of the corresponding
            label data.
        :param label_list: List[str]
            List of label data names.
        """
        super(GroundMotionDatasets, self).__init__()

        # Read the csv file.
        df = pd.read_csv(csv_path)
        self.data_path = df["file_name"]

        labels = df[label_list].values.astype(np.float32)

        # Normalize the label data.
        # Mw, fault distance, and PGA are preprocessed individually.
        temp = labels[:, :3]

        # Normalize to a mean of 0
        temp = temp - np.mean(temp, axis=0, keepdims=True)
        # Divide by the maximum value
        temp = temp / np.max(np.abs(temp), axis=0, keepdims=True)
        # Normalize to a standard deviation of 0.1
        temp = temp / (np.std(temp, axis=0, keepdims=True) / 0.1)

        # Calculate the mean, maximum value, and standard deviation collectively for shallow soil parameters.
        temp_soil = labels[:, 5:]
        temp_soil = temp_soil - np.mean(temp_soil)
        temp_soil = temp_soil / np.max(np.abs(temp_soil))
        temp_soil = temp_soil / (np.std(temp_soil) / 0.1)

        # Calculate the mean, maximum value, and standard deviation collectively for deep sedimentary layer parameters.
        temp_deep = labels[:, 3:5]
        temp_deep = temp_deep - np.mean(temp_deep)
        temp_deep = temp_deep / np.max(np.abs(temp_deep))
        temp_deep = temp_deep / (np.std(temp_deep) / 0.1)

        # Combine the processed label data.
        out_label = np.concatenate([temp, temp_deep, temp_soil], axis=1)
        self.label = torch.from_numpy(out_label).clone()

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]
        temp_mat = np.load(path, allow_pickle=True)
        out_tensor = torch.from_numpy(temp_mat.astype(np.float32)).clone()
        out_tensor = out_tensor.reshape(1, 1, -1)

        out_label = self.label[index, :]

        return out_tensor, out_label


class MappingNetwork(nn.Module):
    """
    Mapping network of styleGAN2 based model.
    """

    def __init__(self, z_dim: int, w_dim: int, start_size: int = 16, end_size: int = None) -> None:
        """
        Parameters
        ----------
        z_dim : int
            Dimension of latent vector.
        w_dim : int
            Dimension of intermediate latent vector.
        start_size : int, optional
            Size of the input constant, (b, ??, 1, start_size), by default 16.
        end_size : int, optional
            Size of the generated wave, (b, 1, 1, end_size), by default None.
        """
        super(MappingNetwork, self).__init__()

        # Confirm end_size is appropriate.
        if end_size is None:
            raise ValueError("end_size must be specified.")

        self.style_num = int(np.log2(end_size / start_size)) * 2 + 2

        self.model = nn.Sequential(
            # 1st layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=z_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=z_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 2nd layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=z_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=z_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 3rd layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=z_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=z_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 4th layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=z_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=z_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 5th layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=z_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=z_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 6th layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=z_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=z_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 7th layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=z_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=z_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
            # 8th layer
            model.EqualizedLinear(in_dim=z_dim, out_dim=w_dim, lr_mul=0.01),
            model.AddBiasChannelWise(out_c=w_dim, bias_scale=0.01),
            model.Amplify(rate=2**0.5),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = model.noise_normalization(x)
        x = self.model(x)

        batch_size_c, vector_len_c = x.shape
        x = x.view(batch_size_c, 1, vector_len_c).expand(batch_size_c, self.style_num, vector_len_c)

        return x


class LabelPredNetwork(nn.Module):
    """
    Network for predicting the label data.
    It takes the feature map just before the output layer of the SynthesisNetwork as input and
    outputs label data corresponding to the generated ground motion.
    """

    def __init__(self, in_c: int, label_dim: int) -> None:
        """
        Parameters
        ----------
        :param in_c: Dimension of input feature map (Flatten).
        :param label_dim:  Dimension of output label data.
        """
        super(LabelPredNetwork, self).__init__()

        self.model = nn.Sequential(
            # 1st layer
            model.EqualizedLinear(in_dim=in_c, out_dim=4096 * 2, lr_mul=1),
            model.AddBiasChannelWise(out_c=4096 * 2, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 2nd layer
            model.EqualizedLinear(in_dim=4096 * 2, out_dim=2048, lr_mul=1),
            model.AddBiasChannelWise(out_c=2048, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 3rd layer
            model.EqualizedLinear(in_dim=2048, out_dim=512, lr_mul=1),
            model.AddBiasChannelWise(out_c=512, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 4
            model.EqualizedLinear(in_dim=512, out_dim=256, lr_mul=1),
            model.AddBiasChannelWise(out_c=256, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 5
            model.EqualizedLinear(in_dim=256, out_dim=128, lr_mul=1),
            model.AddBiasChannelWise(out_c=128, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 6
            model.EqualizedLinear(in_dim=128, out_dim=64, lr_mul=1),
            model.AddBiasChannelWise(out_c=64, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 7
            model.EqualizedLinear(in_dim=64, out_dim=32, lr_mul=1),
            model.AddBiasChannelWise(out_c=32, bias_scale=1),
            nn.LeakyReLU(negative_slope=0.2),
            # 8
            model.EqualizedLinear(in_dim=32, out_dim=label_dim, lr_mul=1),
            model.AddBiasChannelWise(out_c=label_dim, bias_scale=1),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        out = self.model(fmap)

        return out


class SynthesisNetwork(nn.Module):
    """
    Synthesis network of styleGAN2 based model.
    """

    def __init__(self, w_dim: int, label_dim: int, alpha_elu: float = 1.0) -> None:
        """
        Parameters
        ----------
        :param w_dim: Dimension of intermediate latent vector.
        :param label_dim: Dimension of output label data.
        :param alpha_elu: Hyperparameter of ELU activation function.
        """
        super(SynthesisNetwork, self).__init__()

        self.const_input = nn.Parameter(torch.randn(1, 512, 1, 16))
        self.label_pred = LabelPredNetwork(in_c=4 * 8192, label_dim=label_dim)

        self.normal_conv_16 = nn.Sequential(
            model.ModulateConv2d(in_c=512, out_c=512, w_dim=w_dim, kw=3, kh=1, stride=1, pad=1),
            model.PixelWiseNoise(wave_width=16),
            model.AddBiasChannelWise(out_c=512, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_16_to_32 = nn.Sequential(
            model.UpSampleConv2d(in_c=512, out_c=512, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=512),
            model.PixelWiseNoise(wave_width=32),
            model.AddBiasChannelWise(out_c=512, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_32 = nn.Sequential(
            model.ModulateConv2d(in_c=512, out_c=512, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=32),
            model.AddBiasChannelWise(out_c=512, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_32_to_64 = nn.Sequential(
            model.UpSampleConv2d(in_c=512, out_c=512, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=512),
            model.PixelWiseNoise(wave_width=64),
            model.AddBiasChannelWise(out_c=512, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_64 = nn.Sequential(
            model.ModulateConv2d(in_c=512, out_c=512, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=64),
            model.AddBiasChannelWise(out_c=512, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_64_to_128 = nn.Sequential(
            model.UpSampleConv2d(in_c=512, out_c=256, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=256),
            model.PixelWiseNoise(wave_width=128),
            model.AddBiasChannelWise(out_c=256, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_128 = nn.Sequential(
            model.ModulateConv2d(in_c=256, out_c=256, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=128),
            model.AddBiasChannelWise(out_c=256, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_128_to_256 = nn.Sequential(
            model.UpSampleConv2d(in_c=256, out_c=128, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=128),
            model.PixelWiseNoise(wave_width=256),
            model.AddBiasChannelWise(out_c=128, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_256 = nn.Sequential(
            model.ModulateConv2d(in_c=128, out_c=128, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=256),
            model.AddBiasChannelWise(out_c=128, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_256_to_512 = nn.Sequential(
            model.UpSampleConv2d(in_c=128, out_c=64, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=64),
            model.PixelWiseNoise(wave_width=512),
            model.AddBiasChannelWise(out_c=64, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_512 = nn.Sequential(
            model.ModulateConv2d(in_c=64, out_c=64, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=512),
            model.AddBiasChannelWise(out_c=64, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_512_to_1024 = nn.Sequential(
            model.UpSampleConv2d(in_c=64, out_c=32, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=32),
            model.PixelWiseNoise(wave_width=1024),
            model.AddBiasChannelWise(out_c=32, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_1024 = nn.Sequential(
            model.ModulateConv2d(in_c=32, out_c=32, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=1024),
            model.AddBiasChannelWise(out_c=32, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_1024_to_2048 = nn.Sequential(
            model.UpSampleConv2d(in_c=32, out_c=16, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=16),
            model.PixelWiseNoise(wave_width=2048),
            model.AddBiasChannelWise(out_c=16, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_2048 = nn.Sequential(
            model.ModulateConv2d(in_c=16, out_c=16, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=2048),
            model.AddBiasChannelWise(out_c=16, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_2048_to_4096 = nn.Sequential(
            model.UpSampleConv2d(in_c=16, out_c=8, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=8),
            model.PixelWiseNoise(wave_width=4096),
            model.AddBiasChannelWise(out_c=8, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.normal_conv_4096 = nn.Sequential(
            model.ModulateConv2d(in_c=8, out_c=8, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=4096),
            model.AddBiasChannelWise(out_c=8, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.ELU(alpha=alpha_elu),
        )

        self.up_sample_conv_4096_to_8192 = nn.Sequential(
            model.UpSampleConv2d(in_c=8, out_c=4, kh=1, kw=3, pad=0, stride=2, w_dim=w_dim),
            model.BlurPooling(in_c=4),
            model.PixelWiseNoise(wave_width=8192),
            model.AddBiasChannelWise(out_c=4, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.normal_conv_8192 = nn.Sequential(
            model.ModulateConv2d(in_c=4, out_c=4, kh=1, kw=3, stride=1, pad=1, w_dim=w_dim),
            model.PixelWiseNoise(wave_width=8192),
            model.AddBiasChannelWise(out_c=4, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.to_wave_16 = nn.Sequential(
            model.ModulateConv2d(in_c=512, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_32 = nn.Sequential(
            model.ModulateConv2d(in_c=512, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_64 = nn.Sequential(
            model.ModulateConv2d(in_c=512, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_128 = nn.Sequential(
            model.ModulateConv2d(in_c=256, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_256 = nn.Sequential(
            model.ModulateConv2d(in_c=128, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_512 = nn.Sequential(
            model.ModulateConv2d(in_c=64, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_1024 = nn.Sequential(
            model.ModulateConv2d(in_c=32, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_2048 = nn.Sequential(
            model.ModulateConv2d(in_c=16, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_4096 = nn.Sequential(
            model.ModulateConv2d(in_c=8, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.to_wave_8192 = nn.Sequential(
            model.ModulateConv2d(in_c=4, out_c=1, kh=1, kw=1, stride=1, pad=0, w_dim=w_dim, demodulate=False),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.main_net_list = [
            self.up_sample_conv_16_to_32,
            self.normal_conv_32,
            self.up_sample_conv_32_to_64,
            self.normal_conv_64,
            self.up_sample_conv_64_to_128,
            self.normal_conv_128,
            self.up_sample_conv_128_to_256,
            self.normal_conv_256,
            self.up_sample_conv_256_to_512,
            self.normal_conv_512,
            self.up_sample_conv_512_to_1024,
            self.normal_conv_1024,
            self.up_sample_conv_1024_to_2048,
            self.normal_conv_2048,
            self.up_sample_conv_2048_to_4096,
            self.normal_conv_4096,
            self.up_sample_conv_4096_to_8192,
            self.normal_conv_8192,
        ]
        self.to_wave_list = [
            self.to_wave_32,
            self.to_wave_64,
            self.to_wave_128,
            self.to_wave_256,
            self.to_wave_512,
            self.to_wave_1024,
            self.to_wave_2048,
            self.to_wave_4096,
            self.to_wave_8192,
        ]

        self.up_sample_wave = model.UpSampleWave(in_c=1)

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b_size, style_num, w_dim = w.shape

        # Replicate the const_input for the number of batch size.
        b_const_input = self.const_input.repeat(b_size, 1, 1, 1)

        # Output of the 1st layer.
        f_map = self.normal_conv_16([b_const_input, w[:, 0]])
        out_wave = self.to_wave_16([f_map, w[:, 1]])
        # skip_wave = None

        # print(f_map.shape)
        # print(out_wave.shape)
        # print(skip_wave.shape)

        for i in range(len(self.to_wave_list)):
            f_map = self.main_net_list[i * 2]([f_map, w[:, i * 2 + 1]])
            # print(f_map.shape)
            f_map = self.main_net_list[i * 2 + 1]([f_map, w[:, i * 2 + 2]])
            # print(f_map.shape)
            skip_wave = self.up_sample_wave(out_wave)
            # print(skip_wave.shape)
            out_wave = self.to_wave_list[i]([f_map, w[:, i * 2 + 3]]) + skip_wave

        # Prediction of the label data.
        out_label = self.label_pred(f_map.reshape(-1, 8192 * 4))

        return out_wave, out_label


class Generator(nn.Module):
    """
    Generator of styleGAN2 based model.
    """

    def __init__(self, z_dim: int, w_dim: int, label_dim: int, wave_len: int, style_mixing_prob: float = 0.9) -> None:
        """
        Parameters
        ----------
        :param z_dim:                Dimension of latent vector.
        :param w_dim:                Dimension of intermediate latent vector.
        :param label_dim:            Dimension of output label data.
        :param wave_len:             Length of the generated wave.
        :param style_mixing_prob:    Probability of style mixing.
        """
        super(Generator, self).__init__()

        self.mapping_network = MappingNetwork(z_dim=z_dim, w_dim=w_dim, end_size=wave_len)
        self.synthesis_network = SynthesisNetwork(w_dim=w_dim, label_dim=label_dim)
        self.style_mixing_prob = style_mixing_prob
        self.style_const_num = self.mapping_network.style_num

        self.z_dim = z_dim

    def forward(self, z: torch.Tensor, is_train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.mapping_network(z)
        s = s1

        if is_train:
            # mixing regularization
            temp = np.random.uniform(low=0, high=1)
            if temp < self.style_mixing_prob:
                z2 = torch.randn(size=(s1.shape[0], self.z_dim)).to(device)
                s2 = self.mapping_network(z2)
                mix_index = np.random.randint(low=0, high=self.style_const_num)

                s = torch.cat([s1[:, :mix_index, :], s2[:, mix_index:, :]], dim=1)
            else:
                pass
        else:
            pass

        out_wave, out_label = self.synthesis_network(s)
        return out_wave, s, out_label


class Discriminator(nn.Module):
    """
    Discriminator of styleGAN2 based model.
    """

    def __init__(self, label_dim: int) -> None:
        """
        Parameters
        ----------
        :param label_dim: Dimension of the label data.
        """
        super(Discriminator, self).__init__()
        # self.label_number = label_number

        self.from_wave = model.FromWave(in_c=1, out_c=16, lr_mul=1.0)

        # Residual blocks
        self.res_block_1 = model.ResBlockDiscriminator(in_c=16, out_c=32)  # 1x8192 -> 1x4096
        self.res_block_2 = model.ResBlockDiscriminator(in_c=32, out_c=64)  # 1x4096 -> 1x2048
        self.res_block_3 = model.ResBlockDiscriminator(in_c=64, out_c=128)  # 1x2048 -> 1x1024
        self.res_block_4 = model.ResBlockDiscriminator(in_c=128, out_c=256)  # 1x1024 -> 1x512
        self.res_block_5 = model.ResBlockDiscriminator(in_c=256, out_c=512)  # 1x512  -> 1x256
        self.res_block_6 = model.ResBlockDiscriminator(in_c=512, out_c=512)  # 1x256  -> 1x128
        self.res_block_7 = model.ResBlockDiscriminator(in_c=512, out_c=512)  # 1x128  -> 1x64
        self.res_block_8 = model.ResBlockDiscriminator(in_c=512, out_c=512)  # 1x64   -> 1x32
        self.res_block_9 = model.ResBlockDiscriminator(in_c=512, out_c=512)  # 1x32   -> 1x16

        # At this stage, the data size = [b, 512, 1, 16]

        self.model_final = nn.Sequential(
            model.MiniBatchStdDev(group_size=4, num_features=1),
            model.Conv2dLayer(in_c=513, out_c=512, kh=1, kw=3, pad=1, stride=1, lr_mul=1.0),
            model.AddBiasChannelWise(out_c=512, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.model_final_dense = nn.Sequential(
            model.EqualizedLinear(in_dim=512 * 16, out_dim=512, lr_mul=1.0),
            model.AddBiasChannelWise(out_c=512, bias_scale=1.0),
            model.Amplify(rate=np.sqrt(2)),
            nn.LeakyReLU(negative_slope=0.2),
            model.EqualizedLinear(in_dim=512, out_dim=1, lr_mul=1.0),
            model.AddBiasChannelWise(out_c=1, bias_scale=1.0),
        )

        self.embedding = nn.Sequential(
            # 1
            model.EqualizedLinear(in_dim=label_dim, out_dim=128, lr_mul=1.0),
            model.AddBiasChannelWise(out_c=128, bias_scale=1.0),
            nn.LeakyReLU(negative_slope=0.2),
            # 2
            model.EqualizedLinear(in_dim=128, out_dim=1024, lr_mul=1.0),
            model.AddBiasChannelWise(out_c=1024, bias_scale=1.0),
            nn.LeakyReLU(negative_slope=0.2),
            # 3
            model.EqualizedLinear(in_dim=1024, out_dim=512 * 16, lr_mul=1.0),
            model.AddBiasChannelWise(out_c=512 * 16, bias_scale=1.0),
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch_c = x.shape[0]
        x = self.from_wave(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)
        x = self.res_block_6(x)
        x = self.res_block_7(x)
        x = self.res_block_8(x)
        x = self.res_block_9(x)
        x = self.model_final(x)
        x = x.reshape(batch_c, -1)
        emb_y = self.embedding(label)
        out_y = torch.sum(x * emb_y, dim=1, keepdim=True)
        out = self.model_final_dense(x) + out_y

        return out


if __name__ == "__main__":
    print("initialize")
