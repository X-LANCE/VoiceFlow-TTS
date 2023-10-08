# \[Working in Progress\] VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching
> This is the official implementation of [VoiceFlow](https://arxiv.org/abs/2309.05027).

![traj](resources/traj.png)

## Environment Setup
This repo is tested on **python 3.9** on Linux. You can set up the environment with conda
```shell
# Install required packages
conda create -n vflow python==3.9  # or any name you like
conda activate vflow
pip install -r requirements.txt

# Then, set PATH
source path.sh  # change the env name in it if you don't use "vflow"
```
Note that to avoid the trouble of installing [torchdyn](https://github.com/DiffEqML/torchdyn), we directly copy the torchdyn 1.0.6 version here locally at `torchdyn/`.

The following process may also need `bash` and `perl` commands in your environment.
## Data Preparation

## Training

## Inference

## Acknowledgement
During the development, the following repositories were referred to:
* [Kaldi](https://github.com/kaldi-asr/kaldi) and [UniCATS-CTX-vec2wav](https://github.com/cantabile-kwok/UniCATS-CTX-vec2wav) for most utility scripts in `utils/`.
* [GradTTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS), where most of the model architecture and training pipelines are adopted.
* [VITS](https://github.com/jaywalnut310/vits), whose distributed bucket sampler is used.
* [CFM](https://github.com/atong01/conditional-flow-matching), for the ODE samplers.
## Citation
```
@misc{guo2023voiceflow,
      title={VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching}, 
      author={Yiwei Guo and Chenpeng Du and Ziyang Ma and Xie Chen and Kai Yu},
      year={2023},
      eprint={2309.05027},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
