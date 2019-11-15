#!/usr/bin/python3
# SPDX-License-Identifier: LGPL-2.0
#
# Copyright 2019, Heinrich Schuchardt <xypron.glpk@gmx.de>

"""
Split files into test, validation, train on a per directory basis.
"""

import argparse
import math
import os
import shutil
import sys
import random

def checkEmpty(path):
    """Check that a directory is empty

    If the directory is not empty, an error message is written and the program
    is terminated.

    path - path to be checked
    """
    if 0 < len(os.listdir(path)):
        print("'{}' is not empty".format(path), file=sys.stderr)
        sys.exit(1)

def checkDir(path):
    """Checks if a path relates to a directory.

    If the path does not indicate a directory, an eror message is written and
    the program is terminated.

    x - path to be checked
    """

    if not os.path.isdir(path):
        print("'{}' is not a directory".format(path), file=sys.stderr)
        sys.exit(1)

def split(sourceDir, destDir):
    # let's get a reproducible split
    random.seed(42)
    for level1Dir in os.listdir(sourceDir):
        level1Path = os.path.join(sourceDir, level1Dir)
        if not os.path.isdir(level1Path):
            continue
        for level2Dir in os.listdir(level1Path):
            level2Path = os.path.join(level1Path, level2Dir)
            if not os.path.isdir(level2Path):
                continue
            testPath = os.path.join(destDir, 'test', level1Dir, level2Dir)
            trainPath = os.path.join(destDir, 'train', level1Dir, level2Dir)
            validPath = os.path.join(destDir, 'valid', level1Dir, level2Dir)
            os.makedirs(testPath, exist_ok=True)
            os.makedirs(trainPath, exist_ok=True)
            os.makedirs(validPath, exist_ok=True)
            files = []
            for level3File in os.listdir(level2Path):
                if os.path.isdir(level3File):
                    continue
                files.append(level3File)
            random.shuffle(files)
            size = len(files)
            firstTest = math.floor(0.6 * size)
            firstValid = math.floor(0.8 * size)
            trainFiles = files[:firstTest]
            testFiles = files[firstTest:firstValid]
            validFiles = files[firstValid:]

            for filename in trainFiles:
                shutil.copy(
                    os.path.join(level2Path, filename),
                    os.path.join(trainPath, filename))
            for filename in testFiles:
                shutil.copy(
                    os.path.join(level2Path, filename),
                    os.path.join(testPath, filename))
            for filename in validFiles:
                shutil.copy(
                    os.path.join(level2Path, filename),
                    os.path.join(validPath, filename))

def main():
    """Command line entry point"""

    parser = argparse.ArgumentParser(
        description='Split into test, validation, train')
    parser.add_argument('sourceDir', nargs=1, default='source',
                        help='directory with WAV audio files')
    parser.add_argument('destDir', nargs=1, default='dest',
                        help='directory for spectrogram files')
    args = parser.parse_args()

    checkDir(args.sourceDir[0])
    checkDir(args.destDir[0])
    checkEmpty(args.destDir[0])
    split(args.sourceDir[0], args.destDir[0])

if __name__ == "__main__":
    main()
