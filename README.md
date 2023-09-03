# PRI.M.I.S:
Code for our paper "Privacy Preserving Image Sharing" (PyTorch-based).

This repository contains:
- The Sparse Ternary AutoEncoder which we use to compress images with arbitrary reconstruction quality (as specified by network capacity) and importantly sparse ternary codes to be used for ambiguation purposes. 
- A simple `torch.utils.data.Dataset` class.
- A basic training pipeline to train the network
- A basic inference pipeline to ambiguate/dis-ambiguate image representations.


## Setup and usage:
In a pip environment, simply run `pip install -e .` to install requirements listed under `setup.py`.

Under `primis.data.py` specify the `ROOT_DIR`, where your images are located. You should create a directory `<ROOT_DIR>/splits`, where a text file `all.txt` lists the relative path of all images with respect to the `ROOT_DIR`.

Run `python3 primis/split.py` to randomly specify train, validation and test splits with the desired proportion. This will write `train.txt`, `valid.txt` and `test.txt` under `<ROOT_DIR>/splits/`.

All configurations for training are specified under `experiments.compression.config_train.json`. 
This contains, most importantly, the network settings, as well as the data paths and some general configurations. 
Similarly, `config_infer.py` specifies the configurations for the inference time, most importantly the `train_run`, which is the run-tag specified by tensorboard under `primis/experiments/compression/runs`, where the path to the trained network lies.

Once all configurations are properly set, simply run `python3 experiments/compression/train.py -c <path/to/the/train/config.json>`. You can track the experiments evolution using tensorboard.
Ambiguation and dis-ambiguation are done using `python3 experiments/compression/infer.py <path/to/the/inference/config.json>`.

