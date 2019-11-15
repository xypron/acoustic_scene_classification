#!/usr/bin/python3
# SPDX-License-Identifier: LGPL-2.0
#
# Copyright 2019, Heinrich Schuchardt <xypron.glpk@gmx.de>

"""
This module reads audio files, executes short-time Fourier transforms, and
saves the results as portable network graphics files.

The audio files are assumed to have a filename starting with two characteristic
values followed by a remainder using hyphens as separators. The characteristic
values are used as directory names for the output, e.g.
street_traffic-lyon-1029-42478-a.wav is transformed to
street_traffic/lyon/1029-42478-a.png.
"""

import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import sys

def checkDir(path):
    """Checks if a path relates to a directory.

    If the path does not indicate a directory, an eror message is written and
    the program is terminated.

    x - path to be checked
    """

    if not os.path.isdir(path):
        print("'{}' is not a directory".format(path), file=sys.stderr)
        sys.exit(1)

def convert(sourceDir, destDir):
    """Convert audio files to spectrograms saved as pickle files.

    The audio files are assumed to have a filename starting with two
    characteristic values followed by a remainder using hyphens as
    separators. The characteristic values are used as directory names
    for the output.

    sourceDir - directory with the audio files to convert
    destDir - directory where the pickle files shall be saved
    """

    for filename in os.listdir(sourceDir):
        # Read audio file
        path = os.path.join(sourceDir, filename)
        x, sr = readAudioFile(path)

        # Transform
        x = preEmphasize(x)
        y = transform(x, sr)

        # Save spectrogram as image
        path, extension = os.path.splitext(filename)
        parts = path.split('-', 2)
        path = os.path.join(destDir, parts[0], parts[1])
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, parts[2])
        path += '.png'
        writePngFile(y, path)

def readAudioFile(filename, sr=None):
    """Reads an audio file with default sampling rate 48000Hz

    filename - file to be read
    return - numpy.float32
    """

    x, sr = librosa.load(filename, sr=sr)
    return x, sr

def preEmphasize(x, alpha=.97):
    """emphasize high frequencies

    This filter mixes the signal with its first derivative:

    y[n] = (x[n] - alpha * x[n-1]) / (1 - alpha)

    Reference:
    Haytham Fayek,
    "Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral
    Coefficients (MFCCs) and What's In-Between",
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    The implementation by Haytham Fayek lacks a good first value. This
    implementation uses x[-1] = 2 * x[0] - x[1].
    Further to keep the amplitudes for low frequencies unchanged the output signal
    is divided by (1 - alpha).

    x - original signal
    alpha - proportion of the output signal that comes from the first derivative
    """

    return np.append((1 - 2 * alpha) * x[0] + alpha * x[1],
                     x[1:] - alpha * x[:-1]) / (1 - alpha)

def transform(x, sr):
    """Transform audio to a spectrogram.

    x - audio data as numpy array
    sr - recording rate in Hz
    return - scaled spectrogram as 2D numpy array
    """

    C = librosa.cqt(x, sr=sr, fmin = 10, hop_length=1024, n_bins=256,
                    bins_per_octave=24)
    logC = librosa.power_to_db(np.abs(C))
    scaler = MinMaxScaler(feature_range=(0, 255))
    logC = scaler.fit_transform(logC)
    return logC

def writePngFile(x, filename):
    """Save object as portable network graphics file

    x - object to save
    filename - filename of the pickle file
    """

    img = Image.fromarray(x)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    plt.imsave(filename, img, cmap=plt.cm.gray)

def main():
    """Command line entry point"""

    parser = argparse.ArgumentParser(
        description='Short-time Fourier transform.')
    parser.add_argument('sourceDir', nargs=1, default='source',
                        help='directory with WAV audio files')
    parser.add_argument('destDir', nargs=1, default='dest',
                        help='directory for spectrogram files')
    args = parser.parse_args()

    checkDir(args.sourceDir[0])
    checkDir(args.destDir[0])
    convert(args.sourceDir[0], args.destDir[0])

if __name__ == "__main__":
    main()
