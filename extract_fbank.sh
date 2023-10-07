#!/bin/bash
. ./cmd.sh || exit 1;

nj=16     # number of parallel jobs in feature extraction
sampling_rate=16000        # sampling frequency
fmax=       # maximum frequency
fmin=         # minimum frequency
num_mels=80     # number of mel basis
fft_size=1024   # number of fft points
hop_size=256    # number of shift points
win_length=  # window length

train_set="train" # name of training data directory
dev_set="val"           # name of development data directory
eval_set="val"         # name of evaluation data directory

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
    compute-cmvn-stats.py scp:${featdir}/${feat_name}/${train_set}/feats.scp ${featdir}/${feat_name}/${train_set}/cmvn.ark
    for x in ${train_set} ${dev_set} ${eval_set} ; do
        echo "Applying normalization for dataset ${x}"
        mkdir -p ${featdir}/normed_${feat_name}/${x} ;
        apply-cmvn.py --norm-vars=true --compress True \
                    ${featdir}/${feat_name}/${train_set}/cmvn.ark \
                    scp:${featdir}/${feat_name}/${x}/feats.scp \
                    ark,scp:${featdir}/normed_${feat_name}/${x}/feats.ark,${featdir}/normed_${feat_name}/${x}/feats.scp
    done
fi
