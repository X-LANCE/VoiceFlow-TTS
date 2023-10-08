# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_collate import DistributedBucketSampler
from utils import plot_tensor, save_plot
from model.utils import fix_len_compatibility
import utils


class ModelEmaV2(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        if hasattr(model, "module"):
            self.model_state_dict = deepcopy(model.module.state_dict())
        else:
            self.model_state_dict = deepcopy(model.state_dict())
        self.decay = decay
        self.device = device  # perform ema on different device from model if set

    def _update(self, model, update_fn):
        model_values = model.module.state_dict().values() if hasattr(model, "module") else model.state_dict().values()
        with torch.no_grad():
            for ema_v, model_v in zip(self.model_state_dict.values(), model_values):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model_state_dict


def run(rank, n_gpus, hps):
    logger_text = utils.get_logger(hps.model_dir)
    logger_text.info(hps)
    out_size = fix_len_compatibility(getattr(hps.data, "cut_segment_seconds", 2) * hps.data.sampling_rate // hps.data.hop_length)
    # NOTE: cut_segment_seconds sec of mel-spec

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed + rank)
    torch.cuda.set_device(rank)
    np.random.seed(hps.train.seed + rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    log_dir = hps.model_dir

    if rank == 0:
        print('Initializing logger...')
        logger = SummaryWriter(log_dir=log_dir)
    train_dataset, collate, model = utils.get_correct_class(hps)
    test_dataset, _, _ = utils.get_correct_class(hps, train=False)

    print('Initializing data loaders...')
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    batch_collate = collate
    loader = DataLoader(dataset=train_dataset, shuffle=False, pin_memory=True,
                        collate_fn=batch_collate, batch_sampler=train_sampler,
                        num_workers=4)  # NOTE: if on server, worker can be 4

    print('Initializing model...')
    model = model(**hps.model).to(device)
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams / 1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams / 1e6))
    print('Total parameters: %.2fm' % (model.nparams / 1e6))

    use_gt_dur = getattr(hps.train, "use_gt_dur", False)
    if use_gt_dur:
        print("++++++++++++++> Using ground truth duration for training")

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hps.train.learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=hps.train.test_size)
    for i, item in enumerate(test_batch):
        mel = item['mel']
        if rank == 0:
            logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                             global_step=0, dataformats='HWC')
            save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    try:
        try:
            ckpt = utils.latest_checkpoint_path(hps.model_dir, "EMA_grad_*.pt")
        except IndexError:
            print(f"Cannot find EMA checkpoint. Trying to find normal checkpoint.")
            ckpt = utils.latest_checkpoint_path(hps.model_dir, "grad_*.pt")
        model, optimizer, learning_rate, epoch_logged = utils.load_checkpoint(ckpt, model, optimizer)
        epoch_start = epoch_logged + 1
        print(f"Loaded checkpoint from {epoch_logged} epoch, resuming training.")
        # optimizer.step_num = (epoch_str - 1) * len(train_dataset)
        # optimizer._update_learning_rate()
        global_step = epoch_logged * len(train_dataset)
    except:
        print(f"Cannot find trained checkpoint, begin to train from scratch")
        epoch_start = 1
        global_step = 0
        learning_rate = hps.train.learning_rate
    model = DDP(model, device_ids=[rank])
    ema_model = ModelEmaV2(model, decay=0.9999)  # It's necessary that we put this after loading model.

    print('Start training...')
    iteration = global_step
    for epoch in range(epoch_start, hps.train.n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        fm_losses = []
        mle_losses = []
        with tqdm(loader, total=len(train_dataset) // (n_gpus * hps.train.batch_size)) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['text_padded'].to(device), \
                    batch['input_lengths'].to(device)
                y, y_lengths = batch['mel_padded'].to(device), \
                    batch['output_lengths'].to(device)
                noise = batch['noise_padded']
                if noise is not None:
                    noise = noise.to(device)
                if hps.xvector:
                    spk = batch['xvector'].to(device)
                else:
                    spk = batch['spk_ids'].to(torch.long).to(device)

                dur_loss, prior_loss, fm_loss, l_mle = model(x, x_lengths,
                                                             y, y_lengths,
                                                             noise=noise,
                                                             spk=spk,
                                                             out_size=out_size,
                                                             use_gt_dur=use_gt_dur,
                                                             durs=batch['dur_padded'].to(device) if use_gt_dur else None)
                mle_loss_weight = getattr(hps.train, "mle_loss_weight", 0.0)
                loss = sum([dur_loss, prior_loss, fm_loss, mle_loss_weight * l_mle])

                loss.backward()
                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()
                ema_model.update(model)
                if rank == 0:
                    logger.add_scalar('training/duration_loss', dur_loss.item(),
                                      global_step=iteration)
                    logger.add_scalar('training/prior_loss', prior_loss.item(),
                                      global_step=iteration)
                    logger.add_scalar('training/flow_loss', fm_loss.item(),
                                      global_step=iteration)
                    logger.add_scalar('training/mle_loss', l_mle.item(),
                                      global_step=iteration)
                    logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                      global_step=iteration)
                    logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                      global_step=iteration)

                    dur_losses.append(dur_loss.item())
                    prior_losses.append(prior_loss.item())
                    fm_losses.append(fm_loss.item())

                    if batch_idx % 5 == 0:
                        msg = (f'Epoch: {epoch}, iter: {iteration} | dur_loss: {dur_loss.item():.3f}, prior_loss: {prior_loss.item():.3f}, '
                               f'flow_loss: {fm_loss.item():.3f}, mle loss: {l_mle.item():.3f}')
                        # logger_text.info(msg)
                        progress_bar.set_description(msg)

                iteration += 1
        if rank == 0:
            log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, float(np.mean(dur_losses)))
            log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
            log_msg += '| flow loss = %.3f\n' % np.mean(fm_losses)
            log_msg += '| mle loss = %.3f\n' % np.mean(mle_losses)
            with open(f'{log_dir}/train.log', 'a') as f:
                f.write(log_msg)

            if epoch % hps.train.save_every > 0:
                continue

            model.eval()
            print('Synthesis...')
            with torch.no_grad():
                for i, item in enumerate(test_batch):
                    x = item['phn_ids'].to(torch.long).unsqueeze(0).to(device)
                    if not hps.xvector:
                        spk = item['spk_ids']
                        spk = torch.LongTensor([spk]).to(device)
                    else:
                        spk = item["xvector"]
                        spk = spk.unsqueeze(0).to(device)

                    x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
                    y_enc, y_dec, attn, z, pred_dur = model.module.inference(x, x_lengths, spk=spk, n_timesteps=10, solver="euler")
                    logger.add_image(f'image_{i}/generated_enc',
                                     plot_tensor(y_enc.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/generated_dec',
                                     plot_tensor(y_dec.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    logger.add_image(f'image_{i}/alignment',
                                     plot_tensor(attn.squeeze().cpu()),
                                     global_step=iteration, dataformats='HWC')
                    save_plot(y_enc.squeeze().cpu(),
                              f'{log_dir}/generated_enc_{i}.png')
                    save_plot(y_dec.squeeze().cpu(),
                              f'{log_dir}/generated_dec_{i}.png')
                    save_plot(attn.squeeze().cpu(),
                              f'{log_dir}/alignment_{i}.png')

            utils.save_checkpoint(ema_model, optimizer, learning_rate, epoch, checkpoint_path=f"{log_dir}/EMA_grad_{epoch}.pt")
            utils.save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path=f"{log_dir}/grad_{epoch}.pt")


if __name__ == "__main__":
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    print(f'============> using {n_gpus} GPUS')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8001'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
