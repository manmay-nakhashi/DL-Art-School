import torchaudio
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import os
import sys
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
sys.path.append('/content/DL-Art-School/codes/')
from utils.util import load_model_from_config
from scripts.audio.gen.speech_synthesis_utils import load_speech_dvae, wav_to_mel
file_path = "/data/speech_synth/TTS_datasets/English_US/LJSpeech-1.1/wavs/LJ001-0001.wav"
audio_data, sample_rate = torchaudio.load(file_path)

filter_length = 1024
win_length = 1024
hop_length = 256
sampling_rate = sample_rate
mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=filter_length, hop_length=hop_length,
                                                      win_length=win_length, power=2, normalized=False,
                                                      sample_rate=sampling_rate, f_min=0,
                                                      f_max=8000, n_mels=80,
                                                      norm="slaney")
# Perform Mel Spectrogram
audio_tensor = torch.tensor(audio_data).float()
mel_spectrogram = mel_stft(audio_tensor)
mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
# # Plot Mel Spectrogram

print(audio_data.shape)
dvae = load_speech_dvae()
mel = wav_to_mel(audio_data)
fig, axs = plt.subplots(1, 2, figsize=(40, 6))
im1 = axs[0].imshow(mel.squeeze().numpy(), cmap='inferno', origin='lower', aspect='auto')
axs[0].set_title('Original Mel Spectrogram')
axs[0].set_ylabel('Mel filter')
axs[0].set_xlabel('Time (hop length)')
fig.colorbar(im1, ax=axs[0], format='%+2.0f dB')

codes = dvae.get_codebook_indices(mel)
print(codes)
mel1, mel2 = dvae.decode(codes)
print(mel1.shape)
im2 = axs[1].imshow(mel1.squeeze().detach().numpy(), cmap='inferno', origin='lower', aspect='auto')
axs[1].set_title('Decoded Mel Spectrogram')
axs[1].set_ylabel('Mel filter')
axs[1].set_xlabel('Time (hop length)')
fig.colorbar(im2, ax=axs[1], format='%+2.0f dB')
plt.show()