import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import tempfile

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def convert_to_wav(input_audio_file):
    try:
        audio = AudioSegment.from_file(input_audio_file) # Auto-detects format

        wav_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(wav_temp_file.name, format="wav")
        wav_temp_file.close() 

        print(f"Successfully converted '{input_audio_file}' to WAV: '{wav_temp_file.name}'")
        return wav_temp_file.name

    except Exception as e:
        print(f"Could not convert file '{input_audio_file}' to WAV. Error: {e}")
        print("Ensure FFmpeg is installed and in your system's PATH if the input is not WAV.")
        raise ValueError(f"Audio conversion failed for {input_audio_file}. Is FFmpeg installed?") from e

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
