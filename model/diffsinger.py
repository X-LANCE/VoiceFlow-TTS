import torch.nn as nn
import torch
from torch.nn import Conv1d, Linear
import math
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffSingerNet(nn.Module):
    def __init__(self, in_dims=80, residual_channels=256, encoder_hidden=128, dilation_cycle_length=1, residual_layers=20, spk_emb_dim=192,
                 pe_scale=1000):
        super().__init__()

        self.pe_scale = pe_scale

        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.time_pos_emb = SinusoidalPosEmb(residual_channels)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.spk_mlp = nn.Sequential(
            nn.Linear(spk_emb_dim, spk_emb_dim * 4),
            Mish(),
            nn.Linear(spk_emb_dim * 4, encoder_hidden)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(encoder_hidden, residual_channels, 2 ** (i % dilation_cycle_length))
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, spec_mask, mu, t, spk=None):
        """
        :param spec: [B, M, T]
        :param t: [B, ]
        :param mu: [B, M, T]
        :return:
        """
        # x = spec[:, 0]
        x = spec
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if spk is not None:
            s = self.spk_mlp(spk)
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            cond = s + mu
        else:
            cond = mu

        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, t)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, M, T]
        return x * spec_mask


class DiffSingerNetAddConcat(nn.Module):
    def __init__(self, in_dims=80, residual_channels=256, encoder_hidden=128, dilation_cycle_length=1, residual_layers=20, spk_emb_dim=192,
                 pe_scale=1000):
        # different from Diffsinger, we use concat & add together on spk information.
        # Speaker is concat on time embedding, while added to mu.
        super().__init__()

        self.pe_scale = pe_scale

        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.time_pos_emb = SinusoidalPosEmb(residual_channels)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.spk_mlp = nn.Sequential(
            nn.Linear(spk_emb_dim, spk_emb_dim * 4),
            Mish(),
            nn.Linear(spk_emb_dim * 4, in_dims)
        )
        self.spk_time_mlp = nn.Linear(in_dims + dim, dim)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(encoder_hidden, residual_channels, 2 ** (i % dilation_cycle_length))
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, spec_mask, mu, t, spk=None):
        """
        :param spec: [B, M, T]
        :param t: [B, ]
        :param mu: [B, M, T]
        :return:
        """
        x = spec
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)  # B, t_dim

        if spk is not None:
            s = self.spk_mlp(spk)
            s_repeated = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])  # B, in_dim, L
            cond = s_repeated + mu
            time_spk = torch.cat([t, s], dim=-1)
            t = self.spk_time_mlp(time_spk)
        else:
            cond = mu

        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, t)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, M, T]
        return x * spec_mask
