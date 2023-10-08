import os

import torch
from kaldiio import WriteHelper
from torch.utils.data import DataLoader
from tqdm import tqdm
import tempfile

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tools


def run(rank, n_gpus, hps, args, ckpt, feats_dir, temp_dir):
    logger = tools.get_logger(hps.model_dir, f"inference.{rank}.log")  # NOTE: cannot delete this line.
    device = torch.device('cpu' if not torch.cuda.is_available() else f"cuda:{rank}")
    torch.manual_seed(hps.train.seed)  # NOTE: control seed

    setattr(hps.data, "train_utts" if args.dataset == "train" else "val_utts", f"{temp_dir}/{rank}.txt")

    train_dataset, collate_fn, model = tools.get_correct_class(hps)
    val_dataset, _, _ = tools.get_correct_class(hps, train=False)

    batch_collate = collate_fn
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=4,
        shuffle=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=batch_collate,
        drop_last=True,
        num_workers=4,
        shuffle=False,
    )
    model = model(**hps.model).to(device)
    tools.load_checkpoint(ckpt, model, None)
    print(f"Loaded checkpoint from {ckpt}")
    model.eval()
    print(f"Number of parameters: {model.nparams}")
    print(f"Number of encoder parameters: {model.encoder.nparams}")
    print(f"Number of decoder parameters: {model.decoder.nparams}")
    
    if args.dataset == "val":
        which_loader = val_loader  # NOTE: specify the dataset: train or val?
        which_set = val_dataset
    else:
        which_loader = train_loader
        which_set = train_dataset
    
    met = False
    with torch.no_grad():
        with WriteHelper(
            f"ark,scp:{os.getcwd()}/{feats_dir}/feats.{rank}.ark,{feats_dir}/feats.{rank}.scp"
        ) as feats, WriteHelper(
            f"ark,scp:{os.getcwd()}/{feats_dir}/noise.{rank}.ark,{feats_dir}/noise.{rank}.scp"
        ) as noise_feats, open(
            f"{feats_dir}/duration.{rank}", "w"
        ) as duration_writer:
            # NOTE: its necessary to add "os.getcwd" here.
            for batch_idx, batch in tqdm(enumerate(which_loader), total=len(which_loader)):
                utts = batch["utt"]
    
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
    
                x, x_lengths = batch["text_padded"].to(device), batch["input_lengths"].to(
                    device
                )
                dur = batch["dur_padded"].to(device) if args.gt_dur else None
    
                # ================== Decode ======================
                if hps.xvector:
                    if args.use_control_spk:
                        xvector = which_set.spk2xvector[args.control_spk_name]
                        spk = torch.FloatTensor(xvector).squeeze().unsqueeze(0).to(device)
                    else:
                        spk = batch["xvector"].to(device)
                else:
                    if args.use_control_spk:
                        spk = torch.LongTensor([args.control_spk_id]).to(device)
                    else:
                        spk = batch["spk_ids"].to(device)
    
                y_enc, y_dec, attn, z, pred_dur = model.inference(
                    x,
                    x_lengths,
                    n_timesteps=args.timesteps,
                    temperature=1.5,
                    spk=spk,
                    length_scale=1.0,
                    solver=args.solver,
                    gt_dur=dur,
                )
                # =================================================
    
                if args.use_control_spk:
                    save_utt_name = f"[spk_{args.control_spk_name if hps.xvector else args.control_spk_id}]{utts[0]}"
                else:
                    save_utt_name = f"{utts[0]}"
    
                feats(
                    save_utt_name, y_dec.squeeze().cpu().numpy().T
                )  # save to ark and scp, mel: (L, 80)
                noise_feats(save_utt_name, z.squeeze().cpu().numpy().T)
                dur_seq = pred_dur.long().squeeze().cpu().numpy().tolist()
                if isinstance(dur_seq, int):
                    dur_seq = [dur_seq]
                dur_seq = " ".join(list(map(str, dur_seq)))
                duration_writer.write(f"{save_utt_name} {dur_seq}\n")


if __name__ == '__main__':
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    
    n_gpus = torch.cuda.device_count()
    print(f"============> using {n_gpus} GPUS")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "80000"

    hps, args = tools.get_hparams_decode()
    ckpt = tools.latest_checkpoint_path(hps.model_dir, "grad_*.pt" if not args.EMA else "EMA_grad_*.pt")

    if args.use_control_spk:
        feats_dir = f"synthetic_wav/{args.model}/tts_other_spk"
    else:
        feats_dir = f"synthetic_wav/{args.model}/generate_for_reflow/{args.dataset}"
    if not os.path.exists(feats_dir):
        os.makedirs(feats_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Temporary dir is", temp_dir)
        # split hps.data.{dataset}_utts into multiple copies
        which_file_to_split = hps.data.train_utts if args.dataset == "train" else hps.data.val_utts
        with open(which_file_to_split, 'r') as fr:
            lines = fr.readlines()
        total_lines = len(lines)
        lines_per_copy = total_lines // n_gpus
        remaining_lines = total_lines % n_gpus

        for i in range(n_gpus):
            output_file = f"{temp_dir}/{i}.txt"
            with open(output_file, 'w') as fw:
                start = i * lines_per_copy + min(i, remaining_lines)
                end = start + lines_per_copy + (1 if i < remaining_lines else 0)
                fw.writelines(lines[start:end])

        mp.spawn(
            run,
            nprocs=n_gpus,
            args=(
                n_gpus,
                hps,
                args,
                ckpt,
                feats_dir,
                temp_dir
            ),
        )

    with open(f"{feats_dir}/feats.scp", 'w') as output:
        for i in range(n_gpus):
            input_file = f"{feats_dir}/feats.{i}.scp"
            with open(input_file, "r") as fr:
                output.write(fr.read())
    with open(f"{feats_dir}/noise.scp", 'w') as output:
        for i in range(n_gpus):
            input_file = f"{feats_dir}/noise.{i}.scp"
            with open(input_file, "r") as fr:
                output.write(fr.read())
    with open(f"{feats_dir}/duration", 'w') as output:
        for i in range(n_gpus):
            input_file = f"{feats_dir}/duration.{i}"
            with open(input_file, "r") as fr:
                output.write(fr.read())