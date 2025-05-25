from pydub import AudioSegment
import numpy as np
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv
load_dotenv()
threshold = float(os.getenv('threshold'))
chime = os.getenv('chime_file')
file = os.getenv('audio_test')

from utils import match_target_amplitude, graph_spectrogram
from model import prepare_model
model = prepare_model()


def detect_triggerword(file):
    fig, (ax1, ax2) = plt.subplots(2,1)
    audio_clip = AudioSegment.from_wav(file)
    normalized_audio_clip = match_target_amplitude(audio_clip, -20.0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file_for_spectrogram:
        normalized_audio_clip.export(temp_file_for_spectrogram.name, format='wav')
        plt.sca(ax1) # Set current axes to ax1 BEFORE graph_spectrogram plots
        x = graph_spectrogram(temp_file_for_spectrogram.name)
        ax1.set_title("Audio Spectrogram")

    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)

    ax1.set_xlim(0, 50)

    ax2.plot(predictions[0,:,0])
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Time Steps')
    ax2.set_title('Trigger Word Detection Probabilities')
    fig.tight_layout()

    return predictions, fig

def chime_on_activate(input_file, output_file, chime, predictions):
    audio_clip = AudioSegment.from_wav(input_file)
    chime_clip = AudioSegment.from_wav(chime)

    Ty = predictions.shape[1]
    consecutive_timesteps = 0

    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0,i,0] < threshold: consecutive_timesteps = 0
        elif consecutive_timesteps>20:
            print('trigger word detected at', (i/Ty)*audio_clip.duration_seconds, 'seconds')
            audio_clip = audio_clip.overlay(chime_clip, position = (i/Ty)*audio_clip.duration_seconds*1000)
            consecutive_timesteps = 0
    
    audio_clip.export(output_file, format = 'wav')

def model_inference(input_file, chime = False):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    timestamp = int(time.time() * 1000)
    output_file = 'outputs/' + input_file.split('.wav')[0].split('/')[-1] + f'_output_{timestamp}.wav'

    predictions, fig = detect_triggerword(input_file)
    chime_on_activate(input_file, output_file, chime, predictions)
    
    return output_file, fig

if __name__ == '__main__':
    model_inference(file, chime)