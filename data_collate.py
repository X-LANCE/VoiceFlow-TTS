import os.path
import random
import numpy as np
import torch
import re
import torch.utils.data

import kaldiio
from tqdm import tqdm


class BaseCollate:
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def collate_text_mel(self, batch: [dict]):
        """
        :param batch: list of dicts
        This function sorts batch elements by its length and concatenate all batch elements into pytorch tensors
        """
        contains_noise = (batch[0]['noise'] is not None)
        utt = list(map(lambda x: x['utt'], batch))
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['phn_ids']) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]['phn_ids']
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0]['mel'].size(0)
        max_target_len = max([x['mel'].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        noise_padded = torch.zeros_like(mel_padded)
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mel']
            mel_padded[i, :, :mel.size(1)] = mel

            output_lengths[i] = mel.size(1)
            if contains_noise:
                noise = batch[ids_sorted_decreasing[i]]['noise']
                noise_padded[i, :, :mel.size(1)] = noise

        dur_padded = torch.LongTensor(len(batch), max_input_len)
        dur_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]]['dur']
            dur_padded[i, :dur.size(0)] = dur

        utt_name = np.array(utt)[ids_sorted_decreasing].tolist()
        if isinstance(utt_name, str):
            utt_name = [utt_name]

        res = {
            "utt": utt_name,
            "text_padded": text_padded,
            "input_lengths": input_lengths,
            "mel_padded": mel_padded,
            "noise_padded": noise_padded if contains_noise else None,
            "output_lengths": output_lengths,
            "dur_padded": dur_padded
        }
        return res, ids_sorted_decreasing


class SpkIDCollate(BaseCollate):
    def __call__(self, batch, *args, **kwargs):
        base_data, ids_sorted_decreasing = self.collate_text_mel(batch)
        spk_ids = torch.LongTensor(list(map(lambda x: x["spk_ids"], batch)))
        spk_ids = spk_ids[ids_sorted_decreasing]
        base_data.update({
            "spk_ids": spk_ids
        })
        return base_data


class XvectorCollate(BaseCollate):
    def __call__(self, batch, *args, **kwargs):
        base_data, ids_sorted_decreasing = self.collate_text_mel(batch)
        xvectors = torch.cat(list(map(lambda x: x["xvector"].unsqueeze(0), batch)), dim=0)
        xvectors = xvectors[ids_sorted_decreasing]
        base_data.update({
            "xvector": xvectors
        })
        return base_data
 

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
