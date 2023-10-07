import os.path
import random
import numpy as np
import torch
import re
import torch.utils.data

import json

import kaldiio
from tqdm import tqdm


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def check_frame_length(utt, dur, mel):
    assert sum(dur) == mel.shape[1], f"Frame length mismatch: utt {utt}, dur: {sum(dur)}, mel: {mel.shape[1]}"


def check_phone_length(utt, dur, phn):
    assert len(dur) == len(phn), f"Phone length mismatch: utt {utt}, phone length {len(phn)}, dur length {len(dur)}"


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, utts: str, n_mel_channels: int, sampling_rate: int,
                 feats_scp: str, utt2num_frames: str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str, add_blank=False, noise_scp: str = None):
        """
        :param utts: file path. A list of utts for this loader. These are the only utts that this loader has access.
        This loader only deals with text, duration and feats. Other files despite `utts` can be larger.
        :param n_mel_channels: number of mel dimensions
        :param sampling_rate: sampling rate
        :param feats_scp: Kaldi-style feats.scp file path. Should contain contents like "utt1 /path/to/feats.ark:12345"
        :param utt2num_frames: plain text file path, indicating every utterance's number of frames, like "utt1 300"
        :param utt2phns: plain text file path, indicating every utterance's phone sequence, like "utt1 AH0 P AY1 L"
        :param phn2id: plain text file path, indicating every phone's index.
        :param utt2phn_duration: plain text file path, indicating every utterance's duration sequence, like "utt1 10 2 4 8"
        :param add_blank: if True, then insert a <BLANK> token between every phoneme. Useful for MAS.
        :param noise_scp: like feats.scp, but specifying the noise for ReFlow.
        """
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.utts = self.get_utts(utts)
        self.add_blank = add_blank
        self.utt2phn, self.phn2id = self.get_utt2phn(utt2phns, phn2id, add_blank=add_blank)
        self.vocab_len = len(self.phn2id.keys())
        self.utt2phn_dur = self.get_utt2phn_dur(utt2phn_duration)
        self.utt2feat = self.get_utt2feat(feats_scp)
        if noise_scp is not None:
            self.utt2noise = self.get_utt2feat(noise_scp)
        else:
            self.utt2noise = None
        self.utt2num_frames = self.get_utt2num_frames(utt2num_frames)
        self.lengths = [self.utt2num_frames[utt] for utt in self.utts]

    def get_utts(self, utts: str) -> list:
        with open(utts, 'r') as f:
            L = f.readlines()
            L = list(map(lambda x: x.strip(), L))
            random.seed(1234)
            random.shuffle(L)
        return L

    def get_utt2phn(self, utt2phns: str, phn2id: str, add_blank: bool = False) -> (dict, dict):
        res = dict()
        with open(utt2phns, 'r') as f:
            for l in f.readlines():
                res[l.split()[0]] = l.strip().split()[1:]

        res_phn2id = dict()
        with open(phn2id, 'r') as f:
            for l in f.readlines():
                res_phn2id[l.split()[0]] = int(l.strip().split()[1])

        if add_blank:
            blank_id = max(res_phn2id.values()) + 1
            res_phn2id['<BLANK>'] = blank_id

        return res, res_phn2id

    def get_utt2phn_dur(self, utt2phn_duration: str) -> dict:
        res = dict()
        with open(utt2phn_duration, 'r') as f:
            for l in f.readlines():
                uttid = l.split()[0]
                # map to integer
                durs = list(map(int, l.strip().split()[1:]))
                res[uttid] = durs
        return res

    def get_utt2feat(self, feats_scp: str):
        utt2feat = kaldiio.load_scp(feats_scp)  # lazy load mode
        print(f"Succeed reading feats from {feats_scp}")
        return utt2feat

    def get_mel_from_kaldi(self, utt):
        feat = self.utt2feat[utt]
        feat = torch.FloatTensor(np.copy(feat)).squeeze()
        assert self.n_mel_channels in feat.shape
        if feat.shape[0] == self.n_mel_channels:
            return feat
        else:
            return feat.T

    def get_noise_from_kaldi(self, utt):
        feat = self.utt2noise[utt]
        feat = torch.FloatTensor(np.copy(feat)).squeeze()
        assert self.n_mel_channels in feat.shape
        if feat.shape[0] == self.n_mel_channels:
            return feat
        else:
            return feat.T

    def get_text(self, utt):
        phn_seq = self.utt2phn[utt]
        phn_id_seq = list(map(lambda x: self.phn2id[x], phn_seq))
        if self.add_blank:
            phn_id_seq = intersperse(phn_id_seq, self.phn2id['<BLANK>'])
        return torch.LongTensor(phn_id_seq)

    def get_dur_from_kaldi(self, utt):
        return torch.LongTensor(self.utt2phn_dur[utt])

    def get_utt2num_frames(self, utt2num_frames: str):
        res = dict()
        with open(utt2num_frames, 'r') as fr:
            for line in fr.readlines():
                terms = line.strip().split()
                utt, num = terms[0], int(terms[1])
                res[utt] = num
        return res

    def __getitem__(self, index):
        res = self.get_mel_text_pair(self.utts[index])
        return res

    def __len__(self):
        return len(self.utts)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class SpkIDLoader(BaseLoader):
    def __init__(self, utts: str, n_mel_channels: int, sampling_rate: int,
                 feats_scp: str, utt2num_frames:str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str, utt2spk: str, add_blank: bool = False,  noise_scp: str = None):
        """
        :param utt2spk: json file path (utt name -> spk id)
        This loader loads speaker as a speaker ID for embedding table
        """
        super(SpkIDLoader, self).__init__(utts, n_mel_channels, sampling_rate, feats_scp, utt2num_frames, utt2phns, phn2id, utt2phn_duration,
                                          add_blank=add_blank, noise_scp=noise_scp)
        self.utt2spk = self.get_utt2spk(utt2spk)

    def get_utt2spk(self, utt2spk: str) -> dict:
        with open(utt2spk, 'r') as f:
            res = json.load(f)
        return res

    def get_mel_text_pair(self, utt):
        spkid = self.utt2spk[utt]
        phn_ids = self.get_text(utt)
        mel = self.get_mel_from_kaldi(utt)
        T_mel = mel.shape[1]

        if self.utt2noise is not None:
            noise = self.get_noise_from_kaldi(utt)
            T_noise = noise.shape[1]
            if abs(T_mel - T_noise) <= 3:
                min_len = min(T_mel, T_noise)
                mel = mel[:, :min_len]
                noise = noise[:, :min_len]
        else:
            noise = None

        dur = self.get_dur_from_kaldi(utt)
        check_frame_length(utt, dur, mel)
        if not self.add_blank:
            check_phone_length(utt, dur, phn_ids)
        res = {
            "utt": utt,
            "phn_ids": phn_ids,
            "mel": mel,
            "noise": noise,
            "dur": dur,
            "spk_ids": spkid
        }
        return res

    def __getitem__(self, index):
        res = self.get_mel_text_pair(self.utts[index])
        return res

    def __len__(self):
        return len(self.utts)


class XvectorLoader(BaseLoader):
    def __init__(self, utts: str, n_mel_channels: int, sampling_rate: int,
                 feats_scp: str, utt2num_frames: str, utt2phns: str, phn2id: str,
                 utt2phn_duration: str, utt2spk_name: str, spk_xvector_scp: str, add_blank: bool = False, noise_scp: str = None):
        """
        :param utt2spk_name: like kaldi-style utt2spk
        :param spk_xvector_scp: kaldi-style speaker-level xvector.scp
        """
        super(XvectorLoader, self).__init__(utts, n_mel_channels, sampling_rate, feats_scp, utt2num_frames, utt2phns, phn2id, utt2phn_duration,
                                            add_blank=add_blank, noise_scp=noise_scp)
        self.utt2spk = self.get_utt2spk(utt2spk_name)
        self.spk2xvector = self.get_spk2xvector(spk_xvector_scp)

    def get_utt2spk(self, utt2spk):
        res = dict()
        with open(utt2spk, 'r') as f:
            for l in f.readlines():
                res[l.split()[0]] = l.split()[1]
        return res

    def get_spk2xvector(self, spk_xvector_scp: str) -> dict:
        res = kaldiio.load_scp(spk_xvector_scp)
        print(f"Succeed reading xvector from {spk_xvector_scp}")
        return res

    def get_xvector(self, utt):
        xv = self.spk2xvector[self.utt2spk[utt]]
        xv = torch.FloatTensor(xv).squeeze()
        return xv

    def get_mel_text_pair(self, utt):
        phn_ids = self.get_text(utt)
        if self.add_blank:
            phn_ids = intersperse(phn_ids, self.phn2id['<BLANK>'])
        mel = self.get_mel_from_kaldi(utt)
        T_mel = mel.shape[1]

        if self.utt2noise is not None:
            noise = self.get_noise_from_kaldi(utt)
            T_noise = noise.shape[1]
            if abs(T_mel - T_noise) <= 3:
                min_len = min(T_mel, T_noise)
                mel = mel[:, :min_len]
                noise = noise[:, :min_len]
        else:
            noise = None

        dur = self.get_dur_from_kaldi(utt)
        xvector = self.get_xvector(utt)
        check_frame_length(utt, dur, mel)
        if not self.add_blank:
            check_phone_length(utt, dur, phn_ids)
        res = {
            "utt": utt,
            "phn_ids": phn_ids,
            "mel": mel,
            "noise": noise,
            "dur": dur,
            "xvector": xvector,
        }
        return res
