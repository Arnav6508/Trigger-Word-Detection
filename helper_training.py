import numpy as np
import tensorflow as tf
from utils import match_target_amplitude, graph_spectrogram

import os
from dotenv import load_dotenv
load_dotenv()

def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0, high=10000-segment_ms) 
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time

    for previous_start, previous_end in previous_segments:
        if previous_end<segment_start or segment_end<previous_start: continue
        else: return True

    return False


def insert_audio_clip(background, audio_clip, previous_segments):

    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)

    retry = 5 
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry = retry - 1

    if not is_overlapping(segment_time, previous_segments):
        previous_segments.append(segment_time)
        new_background = background.overlay(audio_clip, position = segment_time[0])
    else:
        new_background = background
        segment_time = (10000, 10000)
    
    return new_background, segment_time


def insert_ones(y, segment_end_ms):

    _, Ty = y.shape
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    y[0, segment_end_y+1 : min(segment_end_y + 51, Ty)] = 1 
    return y


def create_training_example(background, activates, negatives, Ty):

    background = background - 20

    y = np.zeros((1, Ty))

    previous_segments = []
 
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    for one_random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, one_random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    background = match_target_amplitude(background, -20.0)

    background.export("train" + ".wav", format="wav")
    x = graph_spectrogram("train.wav")
    
    return x, y

def create_full_training_set(backgrounds, activates, negatives):
    nsamples = os.getenv('nsamples')
    Ty = os.getenv('Ty')
    X, Y = [], []
    for i in range(0, nsamples):
        if i%10 == 0:
            print(i)
        x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
        X.append(x.swapaxes(0,1))
        Y.append(y.swapaxes(0,1))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
