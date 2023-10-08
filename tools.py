# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import argparse
import glob
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import data_loader as loaders
import data_collate as collates
import yaml
from model import GradTTS, GradTTSXvector
import torch


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = 1
    if 'iteration' in checkpoint_dict.keys():
        iteration = checkpoint_dict['iteration']
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    else:
        learning_rate = None
    if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def load_checkpoint_except_decoder(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = 1
    if 'iteration' in checkpoint_dict.keys():
        iteration = checkpoint_dict['iteration']
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    else:
        learning_rate = None
    if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("decoder."):
            new_state_dict[k] = v
            continue
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


def get_correct_class(hps, train=True):
    if train:
        data_hps = hps.data.train
    else:
        data_hps = hps.data.val

    if hps.xvector:
        loader = loaders.XvectorLoader
        collate = collates.XvectorCollate
        model = GradTTSXvector

    else:  # no xvector
        loader = loaders.SpkIDLoader
        collate = collates.SpkIDCollate
        model = GradTTS
    dataset = loader(utts=data_hps.utts,
                     n_mel_channels=hps.data.n_mel_channels,
                     sampling_rate=hps.data.sampling_rate,
                     feats_scp=data_hps.feats_scp,
                     utt2num_frames=data_hps.utt2num_frames,
                     utt2phns=data_hps.utt2phns,
                     phn2id=hps.data.phn2id,
                     utt2phn_duration=data_hps.utt2phn_duration,
                     utt2spk=data_hps.utt2spk,
                     add_blank=hps.data.add_blank,
                     noise_scp=data_hps.noise_scp if hps.perform_reflow else None)

    return dataset, collate(), model


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.yaml",
                        help='YAML file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--seed', type=int, default=1234)

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.yaml")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = yaml.load(data,  Loader=yaml.FullLoader)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.train.seed = args.seed
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)


def get_hparams_decode():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.yaml",
                        help='YAML file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('--dataset', choices=['train', 'val'], default='val', type=str, help='which dataset to use')
    parser.add_argument('--use-control-spk', action='store_true', help='whether to use GT spk or other spk')
    parser.add_argument('--control-spk-id', default=None, type=int, help='if use control spk, then which spk')
    parser.add_argument('--control-spk-name', default=None, type=str, help='if use control spk, then which spk')
    parser.add_argument("--max-utt-num", default=100, type=int, help='maximum utts number to decode')
    parser.add_argument("--specify-utt-name", default=None, type=str, help='if specified, only decodes for that utt')
    parser.add_argument('-t', "--timesteps", type=int, default=10, help='how many timesteps to perform ODE simulation')
    parser.add_argument("--solver", type=str, choices=['rk4', 'euler', 'dopri5', 'tsit5', 'ieuler', 'alf', 'midpoint'], default='euler')

    parser.add_argument("--gt-dur", action="store_true", default=False)
    parser.add_argument("--EMA", action="store_true", default=False)
    parser.add_argument("--duration-scale", type=float, default=0.91, help="Multiplied to predicted duration")
    parser.add_argument("--temperature", type=float, default=0.667, help="Sampling temperature. "
                                                                         "It is used to multiply the Gaussian variance. Lower means stabler.")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.yaml")  # NOTE: which config to load
    with open(config_path, "r") as f:
        data = f.read()
    config = yaml.load(data,  Loader=yaml.FullLoader)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.train.seed = args.seed

    if args.use_control_spk:
        if hparams.xvector:
            assert args.control_spk_name is not None
        else:
            assert args.control_spk_id is not None

    return hparams, args


def get_hparams_decode_outer_text():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.yaml",
                        help='YAML file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('--dataset', choices=['train', 'val'], default='val', type=str, help='which dataset to use')
    parser.add_argument('--use-control-spk', action='store_true', help='whether to use GT spk or other spk')
    parser.add_argument('--control-spk-id', default=None, type=int, help='if use control spk, then which spk')
    parser.add_argument('--control-spk-name', default=None, type=str, help='if use control spk, then which spk')
    parser.add_argument("--max-utt-num", default=100, type=int, help='maximum utts number to decode')
    parser.add_argument("--specify-utt-name", default=None, type=str, help='if specified, only decodes for that utt')
    parser.add_argument('-t', "--timesteps", type=int, default=10, help='how many timesteps to perform reverse diffusion')

    parser.add_argument("--stoc", action='store_true', default=False, help="Whether to add stochastic term into decoding")
    parser.add_argument('--text', type=str, help='text file')
    parser.add_argument('--utt2spk', type=str, help='utt2spk file')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.yaml")  # NOTE: which config to load
    with open(config_path, "r") as f:
        data = f.read()
    config = yaml.load(data,  Loader=yaml.FullLoader)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.train.seed = args.seed

    if args.use_control_spk:
        if hparams.xvector:
            assert args.control_spk_name is not None
        else:
            assert args.control_spk_id is not None

    return hparams, args
