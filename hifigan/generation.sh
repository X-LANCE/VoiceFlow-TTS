#!/bin/bash

. ./cmd.sh

dataset=ljspeech
expdir=exp/train_hifigan.${dataset}

eval_dir=

. parse_options.sh || exit 1;

checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
echo $checkpoint
outdir=$eval_dir/hifigan

# ===========================================
feat-to-len.py scp:${eval_dir}/feats.scp > ${eval_dir}/utt2num_frames || exit 1

mkdir -p ${outdir}/log
echo ========== HifiGAN Generation ==========

${cuda_cmd} --gpu 1 "${outdir}/${name}/log/decode.log" \
    parallel_wavegan/bin/decode.py \
        --feats-scp $eval_dir/feats.scp \
        --num-frames $eval_dir/utt2num_frames \
        --checkpoint "${checkpoint}" \
        --outdir "${outdir}/wav" \
        --verbose "1"
echo "Successfully finished decoding."

