# HifiGAN (parallel_wavegan  implemented version)

We release the trained checkpoints on LJspeech and LibriTTS here.
The detailed information is:

| Dataset  | Sampling Rate | Hop Size | Window Length | Normed |
|----------|---------------|----------|---------------|--------|
| LJSpeech | 16k           | 256      | 1024          | True | 
| LibriTTS | 16k           | 200      | 800           | True |

The trained checkpoint on both datasets are provided online. You can unzip them to sub-folders in `exp/`:
* LJSpeech: [link](https://huggingface.co/cantabile-kwok/hifigan-ljspeech-1024-256/resolve/main/train_hifigan.ljspeech.zip)
* LibriTTS: [link](https://huggingface.co/cantabile-kwok/hifigan-libritts-800-200/resolve/main/train_hifigan.libritts.zip)

Vocoding can be done by 
```shell
cd ../; source path.sh; cd -;  # if path.sh not activated
bash generation.sh --dataset "ljspeech or libritts" --eval_dir /path/that/contains/feats.scp
```
The program will read feats.scp in $eval_dir and synthesize audio to save in that dir.
