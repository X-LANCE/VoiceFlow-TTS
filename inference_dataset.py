import os
import time
import torch
from kaldiio import WriteHelper
from torch.utils.data import DataLoader
from tqdm import tqdm

import tools


def evaluate(hps, args, ckpt, feats_dir):
    logger = tools.get_logger(hps.model_dir, "inference.log")
    device = torch.device('cpu' if not torch.cuda.is_available() else "cuda")
    torch.manual_seed(hps.train.seed)  # NOTE: control seed
    train_dataset, collate_fn, model = tools.get_correct_class(hps)
    val_dataset, _, _ = tools.get_correct_class(hps, train=False)
    batch_collate = collate_fn
    train_loader = DataLoader(dataset=train_dataset, batch_size=1,
                              collate_fn=batch_collate, drop_last=True,
                              num_workers=4, shuffle=False)  # NOTE: if on server, worker can be 4
    val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                            collate_fn=batch_collate, drop_last=True,
                            num_workers=4, shuffle=False)
    model = model(**hps.model).to(device)
    tools.load_checkpoint(ckpt, model, None)
    print(f"Loaded checkpoint from {ckpt}")
    model.eval()
    print(f'Number of parameters: {model.nparams}')
    print(f"Number of encoder parameters: {model.encoder.nparams}")
    print(f"Number of decoder parameters: {model.decoder.nparams}")

    if args.dataset == 'val':
        which_loader = val_loader  # NOTE: specify the dataset: train or val?
        which_set = val_dataset
    else:
        which_loader = train_loader
        which_set = train_dataset

    total_inference_time = 0
    total_inference_frames = 0
    met = False
    with torch.no_grad():
        with WriteHelper(f"ark,scp:{os.getcwd()}/{feats_dir}/feats.ark,{feats_dir}/feats.scp") as feats:
            # NOTE: its necessary to add "os.getcwd" here.
            for batch_idx, batch in tqdm(enumerate(which_loader), total=len(which_loader)):
                utts = batch['utt']
                # ============== Loop Controlling block ============
                if met:
                    break
                if args.specify_utt_name is not None:
                    if not utts[0] == args.specify_utt_name:
                        continue
                    else:
                        met = True
                elif batch_idx >= args.max_utt_num:
                    break
                # ==================================================

                x, x_lengths = batch['text_padded'].to(device), batch['input_lengths'].to(device)
                dur = batch['dur_padded'].to(device) if args.gt_dur else None

                # ================== Decode ======================
                if hps.xvector:
                    if args.use_control_spk:
                        xvector = which_set.spk2xvector[args.control_spk_name]
                        spk = torch.FloatTensor(xvector).squeeze().unsqueeze(0).to(device)
                    else:
                        spk = batch['xvector'].to(device)

                else:
                    if args.use_control_spk:
                        spk = torch.LongTensor([args.control_spk_id]).to(device)
                    else:
                        spk = batch['spk_ids'].to(device)
                s = time.time()
                y_enc, y_dec, attn, z, pred_dur = model.inference(x, x_lengths, n_timesteps=args.timesteps, temperature=args.temperature,
                                                                  spk=spk, length_scale=args.duration_scale, solver=args.solver, gt_dur=dur)
                t = time.time()
                total_inference_time += t - s
                total_inference_frames += y_dec.squeeze().shape[1]
                # =================================================

                if args.use_control_spk:
                    save_utt_name = f"[spk_{args.control_spk_name if hps.xvector else args.control_spk_id}]{utts[0]}"
                else:
                    save_utt_name = f"{utts[0]}"

                feats(save_utt_name, y_dec.squeeze().cpu().numpy().T)  # save to ark and scp, mel: (L, 80)
        print(f"Inference finished. Total time: {total_inference_time}, total frames: {total_inference_frames} "
              f"==> {total_inference_frames / total_inference_time} frame/s")


if __name__ == '__main__':
    hps, args = tools.get_hparams_decode()
    ckpt = tools.latest_checkpoint_path(hps.model_dir, "grad_*.pt" if not args.EMA else "EMA_grad_*.pt")

    if args.use_control_spk:
        feats_dir = f"synthetic_wav/{args.model}/tts_other_spk"
    else:
        feats_dir = f"synthetic_wav/{args.model}/tts_gt_spk"
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)
    evaluate(hps, args, ckpt, feats_dir)
