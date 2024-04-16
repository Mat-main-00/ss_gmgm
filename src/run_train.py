"""
2024. 03. 20.

This script is for training the model.
The hyperparameters used in the training are defined in this script.
Note that the definition of epoch and iteration in the code may differ from the general definition.
    - csv_path:       Path to the CSV file containing information about the training dataset. Please refer to the
                      README for details on the file contents.
    - out_dir:        Path to the directory where training results will be saved, including model parameters and
                      the training loss.
    - save_iter:      Number of iterations to save the model parameters.
    - label_list:     List of condition labels used during training. In this study, the normalization method varies by
                      label, so depending on the problem being addressed, it may be necessary to modify the processing
                      methods in the GroundMotionDatasets class within the model.py file.
    - w_dim:          Dimension of the intermediate latent variable.
    - z_dim:          Dimension of the noise vector.
    - wave_len:       Length of the ground motion data (1D is assumed).
    - num_epoch:      Total number of epochs for training.
    - batch_size:     Batch size.
    - g_train_num:    Number of generator trains per epoch.
    - d_train_num:    Number of discriminator trains per epoch.
    - g_reg_int:      Number of iterations between generator regularization.
    - d_reg_int:      Number of iterations between discriminator regularization.
    - base_lr:        Learning rate.
    - base_beta1:     Beta1 parameter for Adam optimizer.
    - base_beta2:     Beta2 parameter for Adam optimizer.

Note: Be aware that the definitions of "epoch" and "iteration" may differ from general usage within
      the context of this code.
"""
import train

csv_path = "../data/input_file.csv"
out_dir = "../data/out"

label_list = ["mw", "log_fault_dist", "log10_pga", "log10_z1", "log10_z14", "log10_v5", "log10_v10", "log10_v20"]

w_dim = 512
z_dim = 512

wave_len = 8192

num_epoch = 100000
batch_size = 64

g_train_num = 4
d_train_num = 4
g_reg_int = 4
d_reg_int = 16

base_lr = 0.002
base_beta1 = 0.0
base_beta2 = 0.99

save_iter = 5000

# -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    train.train_logistic(
        csv_path=csv_path,
        out_dir=out_dir,
        label_list=label_list,
        z_dim=z_dim,
        w_dim=w_dim,
        wave_len=wave_len,
        num_epoch=num_epoch,
        batch_size=batch_size,
        g_train_num=g_train_num,
        d_train_num=d_train_num,
        g_reg_int=g_reg_int,
        d_reg_int=d_reg_int,
        lr=base_lr,
        beta1=base_beta1,
        beta2=base_beta2,
        save_iter=save_iter,
    )
