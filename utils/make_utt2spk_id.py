#!/usr/bin/env python3

import sys
import json
import os

utt2spk_list = sys.argv[1:]

spks = set()
for file in utt2spk_list:
    with open(file, 'r') as fr:
        for line in fr.readlines():
            spk = line.strip().split()[1]
            # print(spk)
            spks.update([spk])
spk2id = {spk:i for i, spk in enumerate(spks)}
for file in utt2spk_list:
    dirname = os.path.dirname(file)
    utt2id = dict()
    with open(file, 'r') as fr:
        for line in fr.readlines():
            utt, spk = line.strip().split()
            utt2id[utt] = spk2id[spk]
    with open(os.path.join(dirname, "utt2spk_id.json"), 'w') as fw:
        json.dump(utt2id, fw, indent=4, ensure_ascii=False)