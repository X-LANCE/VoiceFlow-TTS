#!/bin/bash
. ./cmd.sh || exit 1;

nj=16     # number of parallel jobs in feature extraction
sampling_rate=16000        # sampling frequency
fmax=       # maximum frequency. If left blank, default to half the sampling rate
fmin=         # minimum frequency. If left blank, default to 0.
num_mels=80     # number of mel basis
fft_size=1024   # number of fft points
hop_size=256    # number of shift points
win_length=  # window length. If left blank, default to minimum integer value that is greater than hop_size and is a power of 2.

train_set="ljspeech/train" # name of training data directory
dev_set="ljspeech/val"           # name of development data directory
eval_set="ljspeech/val"         # name of evaluation data directory

stage=0
stop_stage=100

. parse_options.sh || exit 1;  # This allows you to pass command line arguments, e.g. --fmax 7600
set -eo pipefail

datadir=$PWD/data
featdir=$PWD/feats

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Fbank Feature Extraction"
    for x in ${train_set} ${dev_set} ${eval_set} ; do
        utils/fix_data_dir.sh ${datadir}/${x}
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${sampling_rate} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${fft_size} \
            --n_shift ${hop_size} \
            --win_length "${win_length}" \
            --n_mels ${num_mels} \
            ${datadir}/${x} \
            exp/make_fbank/${x} \
            ${featdir}/fbank/${x}
        mv ${datadir}/${x}/feats.scp ${featdir}/fbank/${x}
    done
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Cepstral Mean Variance Normalization"
    feat_name=fbank

    # if you want to compute the CMVN stats instead of using the provided one, un-comment the line below.
    # compute-cmvn-stats.py scp:${featdir}/${feat_name}/${train_set}/feats.scp ${featdir}/${feat_name}/${train_set}/cmvn.ark
    for x in ${train_set} ${dev_set} ${eval_set} ; do
        echo "Applying normalization for dataset ${x}"
        mkdir -p ${featdir}/normed_${feat_name}/${x} ;
        apply-cmvn.py --norm-vars=true --compress True \
                    ${featdir}/${feat_name}/${train_set}/cmvn.ark \
                    scp:${featdir}/${feat_name}/${x}/feats.scp \
                    ark,scp:${featdir}/normed_${feat_name}/${x}/feats.ark,${featdir}/normed_${feat_name}/${x}/feats.scp
    done
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Write utt2num_frames"
    feat_name=fbank
    for x in ${train_set} ${dev_set} ${eval_set} ; do
        feat-to-len.py scp:${featdir}/normed_${feat_name}/${x}/feats.scp > ${featdir}/normed_${feat_name}/${x}/utt2num_frames
    done
fi

