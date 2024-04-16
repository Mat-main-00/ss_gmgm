"""
2024. 03. 25.

This is a code for generating ground motion data and corresponding condition labels using a model after training.
The generation of data requires Python code that defines the model used for training and a "*.pth" file containing
the model parameters after training. The data saved consist of the following three files:

    1. wave_epoch_*.npy:  Generated ground motion data. This is a 2D numpy array with the shape (noise_num, wave_len).
                          The ground motion data saved in this file are normalized in amplitude.
    2. label_epoch_*.csv: Data of condition labels corresponding to the generated ground motion data.
                          Saved as a csv file. The shape of the data is (noise_num, label_dim).
                          The header contains the names of the condition labels used in the training source file.
    3. input_noise.npy:   An array of noise z used for generating data.
                          This is a 2D numpy array with the shape (noise_num, z_dim).
                          In this code, the same noise is used for each epoch, so only one file is saved.
"""
import os
import sys
import numpy as np
import pandas as pd

import torch

import network
import run_train

# Time step of ground motion data (s)
dt = 0.01

# Number of data to be generated.
noise_num = 100000

# Number of data to generate at once. The larger the number, the faster and more data can be generated,
# but it will consume more memory. Set appropriately according to the amount of available memory.
# Note, this number must be a divisor of "noise_num".
base_num = 100

# Number of epochs for the model generating ground motion data
epoch_list = np.arange(30000, 55000, 5000)

# Random seed
seed = 0

# Directory where the model parameters are saved
model_dir = "../data"

# Directory where the generated data will be saved
save_dir = "../data"

# ######################################################################################################################
if __name__ == "__main__":
    # Check if the directory exists.
    if not os.path.exists(model_dir):
        sys.exit(f"Error: Directory '{model_dir}' does not exist.")

    if not os.path.exists(save_dir):
        sys.exit(f"Error: Directory '{save_dir}' does not exist.")

    # set "device" to use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set the seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # # initialize the noise
    noise_itr_num = noise_num // base_num

    # Load the variables for generating ground motion data
    header = run_train.label_list
    label_dim = len(header)
    z_dim = run_train.z_dim
    w_dim = run_train.w_dim
    wave_len = run_train.wave_len

    # Noise used for generating data
    if os.path.exists(os.path.join(save_dir, 'input_noise.npy')):
        print("Load the noise array.")
        fixed_noise = np.load(os.path.join(save_dir, 'input_noise.npy'))
        fixed_noise = torch.from_numpy(fixed_noise).clone()
        fixed_noise = fixed_noise.to(device)

        # check noise shape
        if fixed_noise.shape[0] != noise_num or fixed_noise.shape[1] != z_dim:
            raise ValueError("The shape of the noise array is incorrect.")
    else:
        print("Create a new noise array.")
        fixed_noise = torch.randn(size=(noise_num, z_dim), device=device)
        save_noise_mat = fixed_noise.to('cpu').detach().numpy()
        # Save noise
        np.save(os.path.join(save_dir, 'input_noise.npy'), save_noise_mat)

    # Generate ground motion data
    for epoch in epoch_list:
        netG = network.Generator(z_dim=z_dim, w_dim=w_dim, label_dim=label_dim, wave_len=wave_len).to(device)

        model_path = os.path.join(model_dir, 'model_G_epoch_{}.pth'.format(epoch))
        state_dict = torch.load(model_path)
        netG.load_state_dict(state_dict)
        netG.eval()

        save_wave_mat = np.zeros((noise_num, wave_len))
        save_label_mat = np.zeros((noise_num, label_dim))

        for itr in range(noise_itr_num):
            noise = fixed_noise[itr * 100: (itr + 1) * 100, :]
            # noise = fixed_noise[itr * 100 : (itr + 1) * 100].view(1, 512)
            fake_wave, _, fake_label = netG(noise, is_train=False)
            n_fake_wave = fake_wave.squeeze().to("cpu").detach().numpy()
            n_fake_label = fake_label.squeeze().to("cpu").detach().numpy()
            save_wave_mat[itr * base_num: (itr + 1) * base_num, :] = n_fake_wave
            save_label_mat[itr * base_num: (itr + 1) * base_num, :] = n_fake_label

            if (itr + 1) % 10 == 0:
                print(
                    "Progress: [EPOCH: {}] [{} / {}]".format(
                        epoch, itr + 1, 1000
                    )
                )

        save_label_df = pd.DataFrame(save_label_mat, columns=header)

        # save the data
        np.save(os.path.join(save_dir, "wave_epoch_{}.npy".format(epoch)), save_wave_mat)
        save_label_df.to_csv(os.path.join(save_dir, "label_epoch_{}.csv".format(epoch)), index=False)
