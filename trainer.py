'''
Created by Quincy Els
2/28/2021
MUSC4611 Capstone Project
Northeastern University

This file contains the script used for extracting musical features and creating a database given a bank of song files.
Originally used against GITZAN data set.

NOTE: jazz.00054.wav was excluded from data set(corrupted multiple times)
'''
import csv
import pathlib
import platform
import matplotlib.pyplot as plt
import librosa.display

import warnings

import numpy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Remove warnings from matplotlib upgrade
warnings.filterwarnings('ignore')

# Librosa Configuration Values (May need to be changed if songs changed)
sr = None  # Resampling rate for songs, set to None to disable
x_axis = 'time'  # X Axis unit for graphs
y_axis = 'log'  # Y Axis unit for graphs, e.g. 'hz' or 'log'
mono = True  # Set to false for stereo tracks
window_size = 1024
hop_length = 512
song_duration = 30  # Song duration in seconds (makes it run quicker)

# Get OS & Current Location
WORKING_DIR = str(pathlib.Path().absolute())
FILE_DELIM = None
if platform.system() == 'Windows':
    FILE_DELIM = '\\'
else:
    FILE_DELIM = '/'

# Set Default Values for Project
ROOT_AUDIO_PATH = WORKING_DIR + FILE_DELIM + 'raw_data' + FILE_DELIM + 'genres_original' + FILE_DELIM
ROOT_DATA_PATH = WORKING_DIR + FILE_DELIM + 'processed_data' + FILE_DELIM
AUDIO_FILE_EXTENSION = '.wav'
DATA_FILE_EXTENSION = '.txt'
genres = ['blues', 'classical', 'country',
          'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']
number_per_genre = 100

# Set to genre and song number to learn single song, None to learn all
genre = None
song_number = None

# Build Path to Song given Genre and Number
def build_path(song_genre, song_number):
    return ROOT_AUDIO_PATH + song_genre + FILE_DELIM + song_genre + '.' + song_number + AUDIO_FILE_EXTENSION


def learn_song(genre, song_number):
    print("Loading Song: " + build_path(genre, song_number))
    global sr
    x, sr = librosa.load(build_path(genre, song_number), sr=sr, mono=mono, duration=song_duration)

    # === Output Spectograms ===
    window = numpy.hanning(window_size)
    stft = librosa.core.spectrum.stft(x, n_fft=window_size, hop_length=hop_length, window=window)
    out = 2 * numpy.abs(stft) / numpy.sum(window)

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    Xdb = librosa.amplitude_to_db(out, ref=numpy.max)

    p = librosa.display.specshow(Xdb, ax=ax, y_axis=y_axis, x_axis=x_axis)
    fig.savefig(ROOT_DATA_PATH + 'spectogram' + FILE_DELIM + genre + FILE_DELIM + song_number)
    plt.clf()

    # === Extract Features ===
    spectral_centroid = librosa.feature.spectral_centroid(x, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
    bpm = librosa.beat.tempo(x, sr=sr)[0]

    # === Write Features to Mean Data File ===
    file = open(ROOT_DATA_PATH + 'mean_features' + FILE_DELIM + 'features.csv', "a", newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow([genre, song_number, numpy.mean(spectral_centroid),
                         numpy.mean(spectral_rolloff), numpy.mean(zero_crossing_rate), bpm])


if genre is not None and song_number is not None:
    learn_song(genre, song_number)
else:
    for genre in genres:
        for n in range(0, number_per_genre):
            song_number = str(n).zfill(5)
            try:
                learn_song(genre, song_number)
            except:
                print("[ERROR] Unable to Load Song! Genre: " + genre + " Song Number: " + song_number)
