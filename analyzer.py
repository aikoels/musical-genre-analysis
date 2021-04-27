'''
Created by Quincy Els
3/7/2021
MUSC4611 Capstone Project
Northeastern University

Contains the code used to analyze wav files and determine its genre.
'''
import csv
import os
import platform
import pathlib
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings

# Remove warnings from matplotlib upgrade
warnings.filterwarnings('ignore')
# Librosa Configuration Values (Song Analysis Parameters)
import numpy

# Debug Toggle
DEBUG = False
# Toggle Checking the name of the song for the guess to see if it's correct. E.g. Guess is "disco" for "disco.wav"
COUNT_CORRECT = False

sr = None  # Resampling rate for songs, set to None to disable
x_axis = 'time'  # X Axis unit for graphs
y_axis = 'log'  # Y Axis unit for graphs, e.g. 'hz' or 'log'
window_size = 1024
hop_length = 512

# Constants for readability
top_songs = "TOP_SONGS"
most_matches = "MOST_MATCHES"
name = "NAME"
genre = "GENRE"
spectral_rolloff = "SPECTRAL_ROLLOFF"
spectral_centroid = "SPECTRAL_CENTROID"
zero_crossing_rate = "ZERO_CROSSING_RATE"
bpm = "BPM"

# Set feature buffer amounts used for "most_matches" method
feature_buffer = {
    spectral_centroid: .02,  # .02
    spectral_rolloff: .11,  # .11
    zero_crossing_rate: .2,  #
    bpm: .2  #
}

# Chose Comparison Method:
#   most_matches: Select the genre with the most matches, using the feature_buffer values to determine what is a match
#   top_songs: Select a number of the top closest songs which match the input, from those, find the most common genres
COMPARISON_METHOD = most_matches

# System Defaults
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


# Analyze Song & Write Features to Data File
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
    song_spectral_centroid = librosa.feature.spectral_centroid(x, sr=sr)
    song_spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
    song_zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
    song_bpm = librosa.beat.tempo(x, sr=sr)[0]

    # === Write Mean of Features to Data File ===
    file = open(OUTPUT_DIRECTORY + FILE_DELIM + 'features.csv', "a", newline='')
    with file:
        writer = csv.writer(file)
        if DEBUG: print("Spectral Centroid: %f\n"
                        "Spectral Rolloff: %f\n"
                        "Zero Crossing: %f\n"
                        "BPM: %i" % (numpy.mean(
            song_spectral_centroid),
                                     numpy.mean(
                                         song_spectral_rolloff),
                                     numpy.mean(
                                         song_zero_crossing_rate),
                                     song_bpm), end='\n\n')
        writer.writerow([song_name, numpy.mean(song_spectral_centroid),
                         numpy.mean(song_spectral_rolloff), numpy.mean(song_zero_crossing_rate), song_bpm])


def match_songs():
    # === Read Song Features from Data File ===
    file = open(OUTPUT_DIRECTORY + FILE_DELIM + 'features.csv', "r")
    with file:
        csv_reader = csv.reader(file)
        song_data = list(csv_reader)

    # Output Correct Matches
    correct_matches = {}
    for song in song_data:
        matches = {}
        # Open Song Database
        database_file = open(DATA_DIRECTORY + FILE_DELIM + 'mean_features' + FILE_DELIM + 'features.csv', 'r')
        with database_file:
            csv_reader = csv.reader(database_file)
            database_list = list(csv_reader)
            for data_song in database_list:
                if song_match(song, data_song):
                    # If already counting this genre, add one
                    if data_song[0] in matches.keys():
                        matches[data_song[0]] += 1
                    else:  # Otherwise, set the count to 1
                        matches[data_song[0]] = 1
        print("Song Name: %s\nSong Genres:" % song[0], end=' ')
        genres = sorted(matches, key=matches.get)
        genres.reverse()
        # Debug printout of all matches
        if DEBUG:
            print("All Matches:", matches)
        # Attempt to print out genre and subgenre
        try:
            if matches[genres[1]] >= .5 * matches[genres[0]]:  # If the second genre is close, print both
                print(*genres[:2], sep=', ')
            else:
                print(genres[0])
        except:  # Print only genre if getting subgenre fails
            print(genres[0])
        # Count correct guesses
        if COUNT_CORRECT:
            # Correct Genre Guess
            if genres[0] in song[0] and matches[genres[0]] != 1:
                try:  # Add to current tally for genre
                    correct_matches[genres[0]] += 1.0
                except:  # Start new tally for genre
                    correct_matches[genres[0]] = 1
            # Guessed Subgenre
            try:
                if genres[1] in song[0]:
                    try:
                        correct_matches[genres[1]] += .5
                    except:
                        correct_matches[genres[1]] = .5
            except:
                pass
        print()
    if COUNT_CORRECT:
        print(correct_matches)


# Return true if a and b are within (r/100)% of each other
def within(a, b, r):
    return (float(1.0) - float(r)) * float(a) <= float(b) <= (float(1.0) + float(r)) * float(a)


# Returns true if the song is a match in features
def song_match(song_list, data_list):
    data = {name: data_list[0], genre: data_list[1], spectral_centroid: data_list[2], spectral_rolloff: data_list[3],
            zero_crossing_rate: data_list[4], bpm: data_list[5]}

    song = {name: song_list[0], spectral_centroid: song_list[1], spectral_rolloff: song_list[2],
            zero_crossing_rate: song_list[3], bpm: song_list[4]}

    return (within(song[spectral_centroid], data[spectral_centroid], feature_buffer[spectral_centroid]) and
            within(song[spectral_rolloff], data[spectral_rolloff], feature_buffer[spectral_rolloff]) and
            within(song[zero_crossing_rate], data[zero_crossing_rate], feature_buffer[zero_crossing_rate]) and
            within(song[bpm], data[bpm], feature_buffer[bpm]))


clean_output_directory()
for filename in os.listdir(INPUT_DIRECTORY):
    if filename.endswith(".wav"):
        analyze_song(filename)

    else:
        continue

print("\n=====RESULTS=====")
match_songs()
