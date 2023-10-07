""" from https://github.com/jaywalnut310/glow-tts """

import torch
import math
from scipy.stats import betabinom
import numpy as np


def mle_loss(z, m, logs, mask):
    l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m) ** 2))  # neg normal likelihood w/o the constant term
    l = l / torch.sum(torch.ones_like(z) * mask)  # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
    return l


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=0.05):
    # the larger the scaling factor, the more concentrated it is on the diagonal
    # returns a tensor of shape (mel_count, phoneme_count)
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P - 1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def batch_beta_binomial_prior_distribution(phoneme_counts, mel_counts, mel_max_count, scaling_factor=0.05):
    result = torch.zeros(phoneme_counts.max(), mel_max_count)
    for p, m in zip(phoneme_counts.cpu(), mel_counts.cpu()):
        result[:p, :m] = beta_binomial_prior_distribution(p, m, scaling_factor=scaling_factor).transpose(0, 1)
    return result.to(phoneme_counts.device)


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_y_by_max_length(y, y_max_length):
    B, D, L = y.shape
    assert y_max_length >= L
    if y_max_length == L:
        return y
    else:
        new_y = torch.zeros(size=(B, D, y_max_length)).to(y.device)
        new_y[:, :, :L] = y
        return new_y


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2 ** num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0],
                                                                   [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss
