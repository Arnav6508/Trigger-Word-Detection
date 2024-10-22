from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import IPython

from model import prepare_model
from utils import match_target_amplitude, graph_spectrogram

file = 'audio/test1.wav'
chime = 'audio/chime.wav'
model = prepare_model()

def detect_triggerword(file):
    plt.subplot(2,1,1)
    audio_clip = AudioSegment.from_wav(file)
    audio_clip = match_target_amplitude(audio_clip, -20.0)
    file_handle = audio_clip.export('temp.wav', format = 'wav')
    filename = 'temp.wav'

    x = graph_spectrogram(filename)
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    plt.xlim(0, 50)

    plt.subplot(2,1,2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions

def chime_on_activate(file, chime, predictions, threshold):
    audio_clip = AudioSegment.from_wav(file)
    chime_clip = AudioSegment.from_wav(chime)

    Ty = predictions.shape[1]
    consecutive_timesteps = 0

    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0,i,0] < threshold: consecutive_timesteps = 0
        elif consecutive_timesteps>20:
            audio_clip = audio_clip.overlay(chime_clip, position = (i/Ty)*audio_clip.duration_seconds*1000)
            consecutive_timesteps = 0
    
    audio_clip.export('chime_output.wav', format = 'wav')

def main(file, chime):
    predictions = detect_triggerword(file)
    chime_on_activate(file, chime, predictions, 0.5)
    IPython.display.Audio(f'./{file.split('.wav')[0]}_output.wav')

main(file, chime)