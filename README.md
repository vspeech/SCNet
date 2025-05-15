# High-Fidelity GAN-based Vocoder with Conditioning Subband Network and Magnitude-aware Phase Loss 

Recent developments of vocoders are primarily dominated by GAN-based networks targeting to high-quality waveform generation from mel-spectrogram representations. However, these methods typically operate in a black box, which results in a loss of inherent information existing in a mel-spectrogram. In this paper, we propose the SCNet, a GAN-based vocoder with Subband Condition Network to address these limitations. Specifically, SCNet takes a subband signal predicted by a condition network as prior knowledge. Then, the subband signal generates Fourier spectral coefficients by Short-Time Fourier transform (STFT), aiming to integrate into the GAN-based backbone network. Additionally, to avoid the phase wrapping issue, we propose a magnitude-aware anti-wrapping phase loss to compute the instantaneous phase errors between predicted and raw phase values. Meanwhile, the magnitude of raw signal is also incorporated into this loss to achieve more weight where the magnitude is larger. In our experiments, SCNet validates the effectiveness and achieves the superior performance for high quality waveform generation, both on subjective and objective metrics.

## Pre-requisites
1. Python >= 3.10
2. Clone this repository:
```bash
git clone https://github.com/vspeech/SCNet.git
cd SCNet
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```
## Inference
Please refer to the inference.py for details.
```bash
python inference.py 
--input_wavs_dir /path/to/your/input_wav \
--checkpoint_file /path/to/your/cp_scnet/model \
--output_dir /path/to/your/output_wav
```
### Pre-Trained Models
You can download the pre-trained LibriTTS model [here](https://pan.baidu.com/s/13Dk9IuKiss0VM7B-l_PTaQ?pwd=f2si) and copy to cp\_scnet directory.

## References
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [yl4579/HiFTNet](https://github.com/yl4579/HiFTNet)
- [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)
- [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)
