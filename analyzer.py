'''
Created by Quincy Els
3/7/2021
MUSC4611 Capstone Project
Northeastern University

Contains the code used to analyze wav files and determine its genre.
'''
import csv
from csv import reader

import librosa.display
import os
import platform
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings

# Remove warnings from matplotlib upgrade
warnings.filterwarnings('ignore')

# Librosa Configuration Values (Song Analysis Parameters)
import numpy

sr = None  # Resampling rate for songs, set to None to disable
x_axis = 'time'  # X Axis unit for graphs
y_axis = 'log'  # Y Axis unit for graphs, e.g. 'hz' or 'log'
window_size = 1024
hop_length = 512

# Program Defaults
FILE_DELIM = None
if platform.system() == 'Windows':
    FILE_DELIM = '\\'
else:
    FILE_DELIM = '/'
INPUT_DIRECTORY = str(pathlib.Path().absolute()) + FILE_DELIM + "input" + FILE_DELIM
OUTPUT_DIRECTORY = str(pathlib.Path().absolute()) + FILE_DELIM + "output" + FILE_DELIM
DATA_DIRECTORY = str(pathlib.Path().absolute()) + FILE_DELIM + "processed_data" + FILE_DELIM


# Removes Old Outputs from Directory
def clean_output_directory():
    output_files = [f for f in os.listdir(OUTPUT_DIRECTORY) if f.endswith(".png") or f.endswith(".csv")]
    for file in output_files:
        os.remove(os.path.join(OUTPUT_DIRECTORY, file))


def analyze_song(song_name):
    print("Analyzing Song: " + song_name)
    global sr
    x, sr = librosa.load(os.path.join(INPUT_DIRECTORY, song_name), sr=sr)

    # === Output Spectrogram ===
    window = numpy.hanning(window_size)
    stft = librosa.core.spectrum.stft(x, n_fft=window_size, hop_length=hop_length, window=window)
    out = 2 * numpy.abs(stft) / numpy.sum(window)

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    Xdb = librosa.amplitude_to_db(out, ref=numpy.max)

    p = librosa.display.specshow(Xdb, ax=ax, y_axis='log', x_axis='time')
    fig.savefig(OUTPUT_DIRECTORY + song_name.replace('.wav', '').replace('.', '_') + '_spectogram')
    plt.clf()

    # === Extract Features ===
    spectral_centroid = librosa.feature.spectral_centroid(x, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
    bpm = librosa.beat.tempo(x, sr=sr)[0]

    # === Write Mean of Features to Data File ===
    file = open(OUTPUT_DIRECTORY + FILE_DELIM + 'features.csv', "a", newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow([song_name, numpy.mean(spectral_centroid),
                         numpy.mean(spectral_rolloff), numpy.mean(zero_crossing_rate), bpm])

    '''
    Strategy One:
        - Select "closeness buffer" (value for each feature it can be within to match)
            Examples:
                Spectral Centroid - 5%
                Spectral RollOff - 10%
                "Swing" - 3%
        - Iterate through songs, collecting matches
        - Go through matches, determine which genre(s) the most matches come from
        - Experiment with different "closeness buffers"
    Strategy Two:
        - Iterate through database, finding top X matches
        - See which genre(s) the top X matches are from
        - Experiment by changing X
    Experiment By Comparing Strategy One & Two
    '''
    # Open Song Database
    database = open(DATA_DIRECTORY + FILE_DELIM + 'mean_features' + FILE_DELIM + 'features.csv', 'r')
    with database as database_file:
        csv_reader = reader(database_file)
        for song in csv_reader:
            print(song)


clean_output_directory()
for filename in os.listdir(INPUT_DIRECTORY):
    if filename.endswith(".wav"):
        analyze_song(filename)
    else:
        continue
