Acoustic scene classification
=============================

"Darling, where are you?" may sound a bit catchy but it describes well what
acoustic scene classification is about. When interacting with mobile devices we
expect relevant information to be presented with a minimum of input effort.
What is relevant depends on the context in which we are acting. If we are
requesting a route information while sitting at a bus stop we most probably are
looking for directions for travel via a bus, while at a railway station we most
probably are looking for a train connection. One possibility for a device to
identify the context is via geolocation information. But this information may
not be available inside buildings. An alternative approach is the analysis of
ambient sound.

This project demonstrates how a convoluted neural network can be used for
acoustic scene classification.

The main file to look at is the Jupyter notebook
acoustic\_scene\_classification.ipynb

Files
=====

* acoustic\_scene\_classification.ipynb - Jupyter notebook
* acoustic\_scene\_classification.html - HTML export of the Jupyter notebook
* blog/index.html - an overview article
* data/download.sh - a script to download the raw data used
* convert.py - Python script to create log-frequency power spectrograms
* split.py - Python script to split the data into training, validation and test
* train.py - Python script to train the neural network
* predict.py - Python script to make predicitions base on the network

Usage
=====

It is assumed that you are running in a Anaconda environment with the packages
mentioned in the "Software used" chapter installed.

Download the raw data

    cd data/ && ./download.sh && cd ..

Convert the data to spectrograms

    python convert.py data/TAU-urban-acoustic-scenes-2019-development/audio/ \
    data/spectrograms/

Split the data into training, validation, and test sub-sets

    python split.py data/spectrograms/ data/splitted/

To run the Juypter notebook

    jupyter notebook

Train the neural network

    python train.py --epochs 64 data/splitted

After the network is trained a checkpoint file 'checkpoint.pt' is saved. This
file is needed as prediction. By default the file is saved in the current
directory.

The console outputs shows the training progress.

To show all available parameters of train.py you can use

    python train.py --help

The checkpoint can now be used to make predictions by applying the trained model
to a spectrogram:

    python predict data/spectrograms/bus/milan/1115-42136-a.png checkpoint.pt

The console output shows the propabilities of the top five categories.

To show all available parameters of predict.py you can use

    python predict.py --help

Software used
=============

These are the software versions that were used:

* Python 3.7.4
* conda 4.7.12
* ipython 7.8.0
* librosa 0.7.1
* matplotlib 3.1.1
* notebook 6.0.1
* numpy 1.17.2
* pandas 0.25.1
* pillow 6.2.0
* pytorch 1.3.0
* scikit-image 0.15.0
* scikit-learn 0.21.3
* torchvision 0.4.1

License
=======

The code is published under
[LGPL-v2]: https://www.gnu.org/licenses/old-licenses/lgpl-2.0.html

Documentation is published under the
[CC BY-SA 4.0]: http://creativecommons.org/licenses/by-sa/4.0/
license
