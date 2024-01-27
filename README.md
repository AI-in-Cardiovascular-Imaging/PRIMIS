# PRIMIS: PRIvacy-preserving Medical Image Sharing
This repository houses the official PyTorch implementation of our paper titled "**PRIMIS: Privacy-Preserving Medical Image Sharing via Deep Sparsifying Transform Learning with Obfuscation**", Journal of Biomedical Informatics, Elsevier, 2024.

## Overview

Our aim is to design a privacy-preserving data-sharing mechanism that allows medical images to be stored as encoded and obfuscated representations in the public domain without revealing any useful or recoverable content from the images. In tandem, we aim to provide authorized users with compact private keys that could be used to reconstruct the corresponding images. 

Our approach involves utilizing a neural auto-encoder. The convolutional filter outputs are passed through sparsifying transformations to produce multiple compact codes. Each code is responsible for reconstructing different attributes of the image. The key privacy-preserving element in this process is obfuscation through the use of specific pseudo-random noise. When applied to the codes, it becomes computationally infeasible for an attacker to guess the correct representation for all codes, thereby preserving the privacy of the data. 

## **Contents**

- **Sparse Ternary AutoEncoder**: A novel architecture for compressing images, allowing for adjustable reconstruction quality based on network capacity. Produces sparse ternary codes suited for ambiguation processes.
- **Dataset Utility**: A dedicated `torch.utils.data.Dataset` class for streamlined data operations.
- **Training Pipeline**: A structured approach to effectively train the model.
- **Inference Pipeline**: Tools to ambiguate or disambiguate image representations.

<!-- 
This repository contains:
- The Sparse Ternary AutoEncoder which we use to compress images with arbitrary reconstruction quality (as specified by network capacity) and importantly sparse ternary codes to be used for ambiguation purposes. 
- A simple `torch.utils.data.Dataset` class.
- A basic training pipeline to train the network
- A basic inference pipeline to ambiguate/dis-ambiguate image representations.
-->

<!-- 
## Setup and usage:
In a pip environment, simply run `pip install -e .` to install requirements listed under `setup.py`.

Under `primis.data.py` specify the `ROOT_DIR`, where your images are located. You should create a directory `<ROOT_DIR>/splits`, where a text file `all.txt` lists the relative path of all images with respect to the `ROOT_DIR`.

Run `python3 primis/split.py` to randomly specify train, validation and test splits with the desired proportion. This will write `train.txt`, `valid.txt` and `test.txt` under `<ROOT_DIR>/splits/`.
-->


## **Setup & Installation**

1. Set up a pip environment and install the necessary requirements listed under `setup.py` using:
    ```bash
    pip install -e .
    ```
2. Navigate to `primis.data.py` and define the `ROOT_DIR` pointing to your image directory.
3. Inside `<ROOT_DIR>`, create a `splits` directory containing an `all.txt` file. This file should enlist the relative paths of all images with reference to `ROOT_DIR`.
4. Execute the following to determine train, validation, and test splits:
    ```bash
    python3 primis/split.py
    ```

    

## **Configuration**

- **Training**: Define all training-related configurations in `experiments.compression.config_train.json`. This encompasses network parameters, data paths, and other essential configurations.
- **Inference**: For inference, use `config_infer.py`. Ensure to specify the `train_run`, which correlates with the TensorBoard run-tag located at `primis/experiments/compression/runs`. This path leads to the trained model.


## **Training Procedure**

Once configurations are in place, initiate training with:

```bash
python3 experiments/compression/train.py -c <path/to/the/train/config.json>
```

Progress can be monitored via TensorBoard.



## **Inference Procedure**

For ambiguation and disambiguation tasks, execute:

```bash
python3 experiments/compression/infer.py <path/to/the/inference/config.json>
```


<!-- 
All configurations for training are specified under `experiments.compression.config_train.json`. 
This contains, most importantly, the network settings, as well as the data paths and some general configurations. 
Similarly, `config_infer.py` specifies the configurations for the inference time, most importantly the `train_run`, which is the run-tag specified by tensorboard under `primis/experiments/compression/runs`, where the path to the trained network lies.

Once all configurations are properly set, simply run `python3 experiments/compression/train.py -c <path/to/the/train/config.json>`. You can track the experiments evolution using tensorboard.
Ambiguation and dis-ambiguation are done using `python3 experiments/compression/infer.py <path/to/the/inference/config.json>`.
-->

## Citation

If you find our work useful in your research, please consider citing our paper. The bibtex entry is provided below:

```bibtex
@article{shiri2023primis,
  title={PRIMIS: Privacy-Preserving Medical Image Sharing via Deep Sparsifying Transform Learning with Obfuscation},
  author={Shiri, Isaac and Razeghi, Behrooz and Ferdowsi, Sohrab and Salimi, Yazdan and G{\"u}nd{\"u}z, Deniz and Teodoro, Douglas and oloshynovskiy, Slava and Zaidi, Habib},
  journal={Journal of Biomedical Informatics},
  volume={150},
  year={2024},
  publisher={Elsevier}
}
