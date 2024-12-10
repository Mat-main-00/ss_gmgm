# Generation of ground motion data by styleGAN2
This repository contains the code and hyperparameters for the paper:

"Yuma Matsumoto, Taro Yaoyama, Sangwon Lee, Takenori Hida, Tatsuya Itoi; Generative Adversarial Networks‐Based Ground‐Motion Model for Crustal Earthquakes in Japan Considering Detailed Site Conditions. Bulletin of the Seismological Society of America, 2024;; 114 (6): 2886–2911. doi: https://doi.org/10.1785/0120240070.

The list of earthquakes and observation staions in the dataset used for the above study can be found in the `eq_station_info.csv` file in the `data` directory.

Please cite this paper if you use the code in this repository as part of a published research project.

## Operating Environment
The codes in this repository has been tested and is known to run under the following environment:
- Ubuntu 20.04.6 LTS
- conda 22.9.0
- Python 3.8.13
- pytorch 1.13.0
- numpy 1.23.3
- pandas 1.4.4
- matplotlib 3.5.3
 
<!-- Please note that while the code was tested in this specific setup, it is not strictly necessary to have the exact versions listed above to run it successfully.  -->

## Usage
- Code that defines the neural network and code to run the training are located in the `src` directory.
- To perform training, execute `run_train.py`.
- A dataset must be prepared in advance and `input_file.csv` needs to be placed inside the `data` directory before training.
- Data generation using the trained model can be performed by running `generate_data.py`.

### Example structure of `input_file.csv`
For this code, it is necessary to prepare `input_file.csv` for loading ground motion data.
The meanings of the file headers are as follows.

| Header       | Description                                                   |
| ------------ | --------------------------------------------------            |
| `file_name`  | Path to the npy file containing a single ground motion data   |
| `mw`         | Moment magnitude, $M_W$                                       |
| `log_fault_dist` | Natural logarithm of rupture distance, $R_{\mathrm{RUP}}$            |
| `log10_pga`  | Common logarithm of the Peak Ground Acceleration (PGA) of ground motion |
| `log10_v5`   | Common logarithm of $V_{\mathrm{S}5}$                                   |
| `log10_v10`  | Common logarithm of $V_{\mathrm{S}10}$                                  |
| `log10_v20`  | Common logarithm of $V_{\mathrm{S}20}$                                  |
| `log10_z1`   | Common logarithm of $Z_{1.0}$                                           |
| `log10_z14`  | Common logarithm of $Z_{1.4}$                                           |

Each npy file contains a single ground motion data, and the values in each row correspond to the values of the condition labels.
The ground motion data must be normalized in amplitude.

### Structure of `example_*.npy` files
The contents of the example_*.npy files are ground motion time history data in a one-dimensional array, and the length of the data is equal to the number of steps in the ground motion records.

## License
This code is licensed under the MIT License.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
