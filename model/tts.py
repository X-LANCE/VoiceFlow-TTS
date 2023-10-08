# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.cfm import FM, CFM, OTCFM, Wrapper
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility, fix_y_by_max_length, \
    mle_loss, batch_beta_binomial_prior_distribution


class GradTTS(BaseModule):
    def __init__(self, n_vocab=148, n_spks=1, spk_emb_dim=64,
                 n_enc_channels=192, filter_channels=768, filter_channels_dp=256,
                 n_heads=2, n_enc_layers=6, enc_kernel=3, enc_dropout=0.1, window_size=4,
                 n_feats=80, dec_dim=64, sigma_min=0.1, pe_scale=1000,
                 fm_type="CFM", shift_by_mu=False, condition_by_mu=True,
                 fm_net_type="unet", encoder_output_dim=80, **kwargs):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.encoder_output_dim = encoder_output_dim

        self.pe_scale = pe_scale

        self.spk_emb = self.construct_spk_module(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, encoder_output_dim, n_enc_channels,
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size,
                                   spk_emb_dim=spk_emb_dim, n_spks=n_spks)

        self.shift_by_mu = shift_by_mu
        self.condition_by_mu = condition_by_mu
        print(f"shift by mu: {shift_by_mu}; condition by mu: {condition_by_mu}")
        assert (self.shift_by_mu or self.condition_by_mu), "mu at least should participate in some way!"
        print(f"Establishing decoder from {fm_type}")
        self.fm_type = fm_type
        if fm_type == "FM":
            print("======> Note: you specified FM-type decoder. "
                  "If you want to perform ReFlow technique, this is not allowed and you must use CFM decoder.")
            self.decoder = FM(n_feats, dec_dim, spk_emb_dim, sigma_min, pe_scale, net_type=fm_net_type, encoder_output_dim=encoder_output_dim)
        elif fm_type == "CFM":
            self.decoder = CFM(n_feats, dec_dim, spk_emb_dim, sigma_min, pe_scale, shift_by_mu=shift_by_mu, condition_by_mu=condition_by_mu,
                               net_type=fm_net_type, encoder_output_dim=encoder_output_dim)
        elif fm_type == "OTCFM":
            assert (not self.shift_by_mu) and self.condition_by_mu, "In OTCFM decoder, mu must be regarded as condition instead of shift"
            self.decoder = OTCFM(n_feats, dec_dim, spk_emb_dim, sigma_min, pe_scale, method=kwargs['ot_method'],
                                 net_type=fm_net_type, encoder_output_dim=encoder_output_dim)
        else:
            raise NotImplementedError

        if hasattr(self.decoder.estimator, "dim_mults"):
            self.unet_downs = len(self.decoder.estimator.dim_mults) - 1
        else:
            self.unet_downs = 0

        self.binomial_prior_factor = kwargs.get("binomial_prior_factor", None)
        if self.binomial_prior_factor is not None:
            print(f"Using Beta-binomial prior with factor {self.binomial_prior_factor}")

    def construct_spk_module(self, n_spks, spk_emb_dim):
        return torch.nn.Embedding(n_spks, spk_emb_dim)

    @torch.no_grad()
    def inference(self, x, x_lengths, n_timesteps, temperature=1.0, spk=None, length_scale=1.0, solver="dopri5", gt_dur=None):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
            solver: string, for ODE solver.
            gt_dur: if specified, then every phone's predicted duration is overwritten by this duration sequence
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        # Get speaker embedding
        spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        if gt_dur is not None:
            w_ceil = gt_dur.unsqueeze(1)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = torch.randn(size=(encoder_outputs.shape[0], self.n_feats, mu_y.shape[2]), device=mu_y.device) * temperature
        if self.shift_by_mu:
            z = z + mu_y

        decoder_output_traj = self.decoder.inference(z, y_mask, mu_y, n_timesteps, spk, solver=solver)

        decoder_outputs = decoder_output_traj[-1]
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length], z[:, :, :y_max_length], w_ceil

    def forward(self, x, x_lengths, y, y_lengths, noise, spk=None, out_size=None, use_gt_dur=False, durs=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            use_gt_dur: bool
            durs: gt duration
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])  # y: B, 80, L
        spk = self.spk_emb(spk)  # [B, D]

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_max_length = fix_len_compatibility(y_max_length, num_downsamplings_in_unet=self.unet_downs)
        y = fix_y_by_max_length(y, y_max_length)

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if not use_gt_dur:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            # assert (self.fm_type == "CFM") and self.shift_by_mu and (not self.condition_by_mu)
            # traj = self.decoder.backward(y, y_mask, torch.zeros_like(y), n_timesteps=2, spk=spk, solver="euler")
            with torch.no_grad():
                # target = traj[-1]
                MAS_target = y
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                z_square = torch.matmul(factor.transpose(1, 2), MAS_target ** 2)
                z_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), MAS_target)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = z_square - z_mu_double + mu_square + const
                # it's actually the log likelihood of target given the Gaussian with (mu_x, I)

                if self.binomial_prior_factor is not None:
                    beta_prior = batch_beta_binomial_prior_distribution(x_lengths, y_lengths, y_max_length,
                                                                        scaling_factor=self.binomial_prior_factor)
                    eps = 1e-8
                    log_prior = log_prior + torch.log(beta_prior + eps)

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()

                # compute MLE loss
                mu_y_uncut = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)  # here mu_x is not cut.
                l_mle = mle_loss(MAS_target, mu_y_uncut, torch.zeros_like(mu_y_uncut), y_mask)

        else:
            attn = generate_path(durs, attn_mask.squeeze(1)).detach()
            l_mle = torch.tensor(0)

        # Compute loss between predicted log-scaled durations and those obtained from MAS (or GT)
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            out_size = min(out_size, y_max_length)  # if max length < out_size, then decrease out_size
            out_size = -fix_len_compatibility(-out_size, num_downsamplings_in_unet=self.unet_downs)
            # adjust out size by finding the largest multiple of 4 which is smaller than it
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            noise_cut = torch.zeros_like(y_cut)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
                if noise is not None:
                    noise_ = noise[i]
                    noise_cut[i, :, :y_cut_length] = noise_[:, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut  # attn -> [B, text_length, cut_length]. The new alignment path does not begin from top left corner
            y = y_cut
            noise = noise_cut if noise is not None else None
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))  # here mu_x is not cut.
        mu_y = mu_y.transpose(1, 2)  # B, 80, cut_length

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.forward(y, noise, y_mask, mu_y, spk)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        if self.encoder_output_dim == self.n_feats:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = torch.tensor(0.).to(dur_loss.device)
        
        return dur_loss, prior_loss, diff_loss, l_mle

    def voice_conversion(self, y, y_lengths, src_spk, tgt_spk, n_timesteps, solver="dopri5"):
        assert (not self.condition_by_mu) and self.shift_by_mu, \
            ("Currently voice conversion is supported only on mu-as-shift but not mu-as-condition models. "
             "In mu-as-condition models the text must also be provided to do voice conversion")
        y, y_lengths = self.relocate_input([y, y_lengths])
        src_spk = self.spk_emb(src_spk)
        tgt_spk = self.spk_emb(tgt_spk)
        y_max_length = y.shape[-1]
        y_max_length = fix_len_compatibility(y_max_length, num_downsamplings_in_unet=self.unet_downs)
        y = fix_y_by_max_length(y, y_max_length)
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(y.device)
        traj = self.decoder.backward(y, y_mask, torch.zeros_like(y), n_timesteps=n_timesteps, spk=src_spk, solver=solver)
        z = traj[-1]
        mel = self.decoder.inference(z, y_mask, torch.zeros_like(z), n_timesteps=n_timesteps, spk=tgt_spk, solver=solver)
        mel = mel[-1][:, :, :y_max_length]
        return mel

    def compute_likelihood(self, x, x_lengths, y, y_lengths, spk=None, n_timesteps=10, durs=None, solver="euler"):
        with torch.no_grad():
            x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])  # y: B, 80, L
            spk = self.spk_emb(spk)  # [B, D]

            # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
            mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
            y_max_length = y.shape[-1]

            y_max_length = fix_len_compatibility(y_max_length, num_downsamplings_in_unet=self.unet_downs)
            y = fix_y_by_max_length(y, y_max_length)

            y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

            attn = generate_path(durs, attn_mask.squeeze(1)).detach()
            # Align encoded text with mel-spectrogram and get mu_y segment
            mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))  # here mu_x is not cut.
            mu_y = mu_y.transpose(1, 2)  # B, 80, cut_length

        # Compute loss of score-based decoder
        noise_likelihood, total_likelihood, frame_num = self.decoder.compute_likelihood(y, y_mask, mu_y, n_timesteps, spk, solver=solver)

        return noise_likelihood, total_likelihood, frame_num


class GradTTSXvector(GradTTS):
    def __init__(self, n_vocab=148, spk_emb_dim=64,
                 n_enc_channels=192, filter_channels=768, filter_channels_dp=256,
                 n_heads=2, n_enc_layers=6, enc_kernel=3, enc_dropout=0.1, window_size=4,
                 n_feats=80, dec_dim=64, sigma_min=0.1,
                 pe_scale=1000, xvector_dim=512, fm_type="CFM", shift_by_mu=False, condition_by_mu=True, **kwargs):
        super(GradTTSXvector, self).__init__(n_vocab, spk_emb_dim=spk_emb_dim,
                                             n_enc_channels=n_enc_channels, filter_channels=filter_channels, filter_channels_dp=filter_channels_dp,
                                             n_heads=n_heads, n_enc_layers=n_enc_layers, enc_kernel=enc_kernel,
                                             enc_dropout=enc_dropout, window_size=window_size, n_feats=n_feats,
                                             dec_dim=dec_dim, sigma_min=sigma_min, pe_scale=pe_scale,
                                             fm_type=fm_type,
                                             shift_by_mu=shift_by_mu, condition_by_mu=condition_by_mu,
                                             **kwargs)

        self.spk_emb = self.construct_spk_module(xvector_dim, spk_emb_dim)

    def construct_spk_module(self, xvector_dim, spk_emb_dim):
        return torch.nn.Linear(xvector_dim, spk_emb_dim)
