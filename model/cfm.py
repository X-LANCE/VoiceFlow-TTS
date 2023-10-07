import numpy as np
import torch.nn as nn
import torch
from model.diffusion import GradLogPEstimator2d
from model.diffsinger import DiffSingerNet
from torchdyn.core import NeuralODE
from model.optimal_transport import OTPlanSampler
import math
import time
from typing import overload

class Wrapper(nn.Module):
    def __init__(self, vector_field_net, mask, mu, spk):
        super(Wrapper, self).__init__()
        self.net = vector_field_net
        self.mask = mask
        self.mu = mu
        self.spk = spk

    def forward(self, t, x, args):
        # NOTE: args cannot be dropped here. This function signature is strictly required by the NeuralODE class
        t = torch.tensor([t], device=t.device)
        return self.net(x, self.mask, self.mu, t, self.spk)


class FM(nn.Module):
    def __init__(self, n_feats, dim, spk_emb_dim=64, sigma_min: float = 0.1, pe_scale=1000, net_type="unet", encoder_output_dim=80):
        super(FM, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = sigma_min
        self.pe_scale = pe_scale

        print(f"Using flow matching net type: {net_type}")
        if net_type == "unet":
            self.estimator = GradLogPEstimator2d(dim,
                                                 spk_emb_dim=spk_emb_dim,
                                                 pe_scale=pe_scale,
                                                 n_feats=n_feats)
        elif net_type == "diffsinger":
            self.estimator = DiffSingerNet(residual_channels=dim, in_dims=n_feats,
                                           spk_emb_dim=spk_emb_dim, pe_scale=pe_scale,
                                           encoder_hidden=encoder_output_dim)
        else:
            raise NotImplementedError

        self.criterion = torch.nn.MSELoss()

    def ode_wrapper(self, mask, mu, spk):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self.estimator, mask, mu, spk)

    @torch.no_grad()
    def inference(self, z, mask, mu, n_timesteps, spk=None, solver="dopri5"):
        if solver != "xxx":  # FIXME: I wanted to implement Euler on my own but turns out not as good as NeuralODE
            # Build a trajectory
            t_span = torch.linspace(0, 1, n_timesteps+1)  # NOTE: n_timesteps means n+1 points in [0, 1]
            neural_ode = NeuralODE(self.ode_wrapper(mask, mu, spk), solver=solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            x = z
            eval_points, traj = neural_ode(x, t_span)
        else:
            interval = 1 / n_timesteps
            traj = []
            traj.append(z)
            x = z
            for step in range(1, n_timesteps + 1):
                t = torch.tensor([interval * step]).to(x.device)
                vf = self.estimator(x, mask, mu, t, spk)
                x = x + vf * interval
                traj.append(x)
        return traj

    def backward(self, x, mask, mu, n_timesteps, spk=None, solver="euler"):
        t_span = torch.linspace(1, 0, n_timesteps+1)
        neural_ode = NeuralODE(self.ode_wrapper(mask, mu, spk), solver=solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        _, traj = neural_ode(x, t_span)
        return traj

    def compute_likelihood(self, x, mask, mu, n_timesteps, spk=None, solver="euler"):
        device = x.device
        with torch.no_grad():
            back_traj = self.backward(x, mask, mu, n_timesteps, spk=spk, solver=solver).cpu()
            last_sample = back_traj[-1]
            del x  # free cuda memory.
            back_traj = back_traj[:-1].squeeze(1)  # Omit the last one. [timesteps, 80, L]
            time_interval = 1/n_timesteps

            t_span = torch.linspace(1, time_interval, n_timesteps)  # cpu

            D, L = last_sample.shape[1], last_sample.shape[2]  # last sample has shape [1, 80, L]
            # compute its likelihood given N(0,1)
            last_sample_loglike = -0.5 * (last_sample**2).sum() - D*L/2 * math.log(2*math.pi)

        batch_size = 3
        num_runs = 1
        trace_estimate = 0
        for run_index in range(num_runs):
            for start in range(n_timesteps//batch_size+1):
                if start * batch_size == n_timesteps:
                    break
                end_index = min((start+1)*batch_size, n_timesteps)
                back_traj_segment = back_traj[start * batch_size: end_index].to(device)
                back_traj_segment.requires_grad = True
                mu_segment = torch.concat([mu] * (end_index - start * batch_size), 0)
                spk_segment = torch.concat([spk] * (end_index - start * batch_size), 0)
                mask_segment = torch.concat([mask] * (end_index - start * batch_size), 0)
                t_segment = t_span[start * batch_size: end_index].to(device)

                minibatch = (back_traj_segment, mask_segment, mu_segment, t_segment, spk_segment)
                vf = self.estimator(*minibatch)
                del mu_segment, spk_segment, mask_segment, t_segment
                noise = torch.randn_like(back_traj_segment)  # [timesteps, 80, L]
                mult_with_noise = vf * noise  # [timesteps, 80, L]
                mult_with_noise.sum().backward(retain_graph=True)
                grad = back_traj_segment.grad.detach().cpu()
                trace_estimate = trace_estimate - (grad * noise.cpu()).sum().detach().item()
                back_traj_segment.grad = None

                del back_traj_segment, noise
        trace_estimate /= num_runs  # average over runs
        integral_estimate = trace_estimate * time_interval

        estimate_loglike = last_sample_loglike + integral_estimate
        return last_sample_loglike.item(), estimate_loglike.item(), back_traj.shape[-1]

    def forward(self, x1, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x1.shape[0], dtype=x1.dtype, device=x1.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x1, mask, mu, t, spk)

    def loss_t(self, x1, mask, mu, t, spk=None):
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1)
        mu_t = t_unsqueeze * x1
        sigma_t = 1 - (1-self.sigma_min) * t_unsqueeze
        x = mu_t + sigma_t * torch.randn_like(x1)  # sample p_t(x|x_0, x_1)
        ut = (self.sigma_min - 1) / sigma_t * (x - mu_t) + x1

        vector_field_estimation = self.estimator(x, mask, mu, t, spk)
        mse_loss = self.criterion(ut, vector_field_estimation)
        return mse_loss, x

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])


class CFM(FM):
    def __init__(self, n_feats, dim, spk_emb_dim=64, sigma_min: float = 0.1, pe_scale=1000, shift_by_mu=False, condition_by_mu=True,
                 net_type="unet", encoder_output_dim=80):
        super(CFM, self).__init__(n_feats, dim, spk_emb_dim, sigma_min, pe_scale, net_type=net_type, encoder_output_dim=encoder_output_dim)
        self.condition_by_mu = condition_by_mu
        self.shift_by_mu = shift_by_mu

    def sample_x0(self, mu, mask):
        x0 = torch.randn_like(mu)   # N(0,1)
        if self.shift_by_mu:
            x0 = x0 + mu  # N(mu, I)
        mask = mask.bool()
        x0.masked_fill_(~mask, 0)
        return x0

    def forward(self, x1, noise, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(x1.shape[0], dtype=x1.dtype, device=x1.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x1, noise, mask, mu, t, spk)

    def loss_t(self, x1, noise, mask, mu, t, spk=None):
        # construct noise (in CFM theory, this is x0)
        if noise is not None:
            x0 = noise  
        else:
            x0 = self.sample_x0(mu, mask)

        ut = x1 - x0  # conditional vector field.
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1)
        mu_t = t_unsqueeze * x1 + (1 - t_unsqueeze) * x0  # conditional Gaussian mean
        sigma_t = self.sigma_min
        x = mu_t + sigma_t * torch.randn_like(x1)  # sample p_t(x|x_0, x_1)
        if self.condition_by_mu:
            mu_input = mu
        else:
            mu_input = torch.zeros_like(mu)
        vector_field_estimation = self.estimator(x, mask, mu_input, t, spk)
        mse_loss = self.criterion(ut, vector_field_estimation)
        return mse_loss, x

    @torch.no_grad()
    def inference(self, z, mask, mu, n_timesteps, spk=None, solver="dopri5"):
        super_class = super()
        if self.condition_by_mu:
            return super_class.inference(z, mask, mu, n_timesteps, spk=spk, solver=solver)
        else:
            return super_class.inference(z, mask, torch.zeros_like(mu), n_timesteps, spk=spk, solver=solver)


class OTCFM(CFM):
    def __init__(self, n_feats, dim, spk_emb_dim=64, sigma_min: float = 0.1, pe_scale=1000, method="exact",
                 net_type="unet", encoder_output_dim=80):
        raise NotImplementedError("CFM with Optimal Transport Sampling is currently not supported")
        super(OTCFM, self).__init__(n_feats, dim, spk_emb_dim, sigma_min, pe_scale, shift_by_mu=False,
                                    net_type=net_type, encoder_output_dim=encoder_output_dim)
        assert method == 'exact', "OT methods except 'exact' are not considered currently"
        self.ot_sampler = OTPlanSampler(method=method)

    def loss_t(self, x1, noise, mask, mu, t, spk=None):
        # construct noise (in CFM theory, this is x0)
        if noise is not None:
            x0 = noise
        else:
            x0 = self.sample_x0(mu, mask)

        # x1 and x0 shape is [B, 80, L]
        B, D, L = x0.shape
        new_x0 = torch.zeros_like(x1)
        new_x1 = torch.zeros_like(x0)
        for l in range(L):
            sub_x0, sub_x1, i, j = self.ot_sampler.sample_plan_with_index(x1[..., l], x0[..., l])
            index_that_would_sort_i = np.argsort(i)  # To keep i and j synchronized for each position in L
            i = i[index_that_would_sort_i]
            j = j[index_that_would_sort_i]

            new_x0[..., l] = x1[i, :, l]
            new_x1[..., l] = x0[j, :, l]

        x1 = new_x0
        x0 = new_x1

        ut = x1 - x0  # conditional vector field. This is actually x0 - x1 in paper.
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1)
        mu_t = t_unsqueeze * x1 + (1 - t_unsqueeze) * x0  # conditional Gaussian mean
        sigma_t = self.sigma_min
        x = mu_t + sigma_t * torch.randn_like(x1)  # sample p_t(x|x_0, x_1)
        vector_field_estimation = self.estimator(x, mask, mu, t, spk)
        mse_loss = self.criterion(ut, vector_field_estimation)
        return mse_loss, x
