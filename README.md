# SCNet: Enhancing GAN-based Speech Generation with Subband Condition Network and Magnitude-aware Phase Loss

Recent speech generation has been predominantly driven by GAN-based networks aimed at high-quality waveform synthesis from mel-spectrograms. However, these methods often operate as black-box models, leading to the loss of inherent spectral information. In this work, we propose SCNet, a GAN-based vocoder augmented with a Subband Condition Network to address this issue. Specifically, SCNet leverages a subband signal predicted by a lightweight condition network as prior knowledge. This subband signal is then transformed via STFT to obtain Fourier coefficients, which are integrated into the backbone for the enhanced reconstruction. Additionally, to mitigate the phase wrapping, we introduce a magnitude-aware phase loss that computes instantaneous phase errors weighted by the corresponding magnitude, emphasizing regions with higher energy. Experimental results demonstrate that SCNet achieves superior performance in both objective and subjective evaluations for high-quality speech generation.

## Pre-requisites
1. Python >= 3.10
2. Clone this repository:
```bash
git clone https://anonymous.4open.science/r/SCNet-94D1.git
cd SCNet
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```

## Pre-Trained Models
You can download the pre-trained LibriTTS model [here](https://drive.google.com/drive/folders/1Dn8f2PUodjME_SsfkJ8SGEtusXJNvkZI?usp=sharing) and copy to cp\_scnet directory.

## Inference
Please refer to the inference.py for details.
```bash
python inference.py 
--input_wavs_dir /path/to/your/input_wav \
--checkpoint_file /path/to/your/cp_scnet/model \
--output_dir /path/to/your/output_wav
```

## References
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [yl4579/HiFTNet](https://github.com/yl4579/HiFTNet)
- [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)
- [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)
