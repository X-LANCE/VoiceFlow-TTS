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

# Install monotonic_align for MAS
cd model/monotonic_align
python setup.py build_ext --inplace
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

## Easter Eggs & Citation
This repository also contains some experimental functionalities. ⚠️Warning: not guaranteed to be correct!
* **Voice conversion**. As GlowTTS can perform voice conversion via the disentangling property of normalizing flows, it is reasonable that flow matching can also perform it. Method `model.tts.GradTTS.voice_conversion` gives a preliminary try.

* **Likelihood estimation**. Differential equation-based generative models have the ability to estimate data likelihoods by the instantaneous change-of-variable formula
```math
\log p_0(\boldsymbol x(0)) = \log p_1(\boldsymbol  x(1)) + \int _0^1 \nabla_{\boldsymbol x} \cdot {\boldsymbol v}(\boldsymbol x(t), t)\mathrm d t
```
  In practice, integral is replaced by summation, and divergence is replaced by the Skilling-Hutchinson trace estimator. See the Appendix D.2 in [Song, et. al](https://arxiv.org/abs/2011.13456) for theoretical details. I implemented this in `model.tts.GradTTS.compute_likelihood`. 
* **Optimal transport**. The conditional flow matching used in this paper is not a **marginally** optimal transport path but only a **conditionally** optimal path. For the marginal optimal transport, [Tong et. al](https://arxiv.org/abs/2302.00482) introduces to sample $x_0,x_1$ together from the joint optimal transport distribution $\pi(x_0,x_1)$. I tried this in `model.cfm.OTCFM`, though it doe not work very well for now.
* **Different estimator architectures**. You can specify an estimator besides the `GradLogPEstimator2d` by the `model.fm_net_type` configuration. Currently the [DiffSinger](https://ojs.aaai.org/index.php/AAAI/article/view/21350)'s estimator architecture is also supported. You can add more, e.g. that introduced in [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
* 💡**Better alignment learning**. This repo supports supervised duration modeling together with monotonic alignment search as that in GradTTS. However, there might be a better way for MAS in flow-matching TTS. `model.tts.GradTTS.forward` now supports beta binomial prior for alignment maps; and if you want, you can change the variable `MAS_target` to something else, e.g. flow-transformed noise!

Feel free to cite this work if it helps 😄

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
