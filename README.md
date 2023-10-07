# \[Working in Progress\] VoiceFlow: Efficient Text-to-Speech with Rectified Flow Matching
> This is the official implementation of [VoiceFlow](https://arxiv.org/abs/2309.05027).

![traj](resources/traj.png)

## Environment Setup

## Data Preparation

## Training

## Inference

## Acknowledgement
During the development, the following repositories were referred to:
* [Kaldi](https://github.com/kaldi-asr/kaldi), for most utility scripts in `utils/`.
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