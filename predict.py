#!/bin/python3
# SPDX-License-Identifier: LGPL-2.0
#
# Copyright 2019, Heinrich Schuchardt

"""
This module demonstrates the classification of images using a trained model.
"""

import argparse
import json
from PIL import Image, ImageOps
import random
import torch
from torch import nn
from torchvision import transforms

class RandomAudioTransform:
    """Randomly transform short time spectrogram

    The spectrogram is transformed randomly in three ways:

    * The spectrogram frequencies are transposed randomly.
    * The speed is randomly changed.
    * A random sampling window in time is chosen.

    size             - target image size of the spectrogram
    octaves          - number of octaves by which to shift spectrogram
    bins_per_octaves - number of image pixels per octave
    dilation         - maximum factor for changing the speed
    sample_size      - size of the time window in relation to the whole
                       spectrogram
    random           - True: use random values, False: replace random values by
                       fixed values
    """

    def __init__(self, size=224, octaves=.5, bins_per_octave=24, dilation=0.25,
                 sample_size=.5, random=True):
        """Construct new RandomAudioTransform
        """

        self.size = size
        self.octaves = octaves
        self.bins_per_octave = bins_per_octave
        self.dilation = dilation
        self.sample_size = sample_size
        self.random = random

    def rand(self):
        """Generate random number from interval [0., 1.[
        """

        if self.random:
            return random.random()
        return .5

    def __call__(self, img):
        """Transform a spectrogram provided as image

        img    - image to transform
        Return - transformed image
        """

        # Stretch the time axis according sample size and time dilation
        width = int((1. + self.dilation * self.rand())
                    * self.size / self.sample_size)
        img = img.resize(size=[width, img.size[1]], resample=Image.BICUBIC)
        # Take sample from image
        alpha = self.octaves * self.bins_per_octave / (img.size[0] - self.size)
        center = [self.rand(), (1 - alpha) * .5 + alpha * self.rand()]
        img = ImageOps.fit(img, size=[self.size, self.size],
                           method=Image.BICUBIC, centering=center)

        return img

class Predict:
    """Image classifier"""

    def __init__(self):
        """Constructor"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = None
        self.image = None
        self.category_names = None
        self.top_k = 5
        transform_norm = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        self.test_transforms = transforms.Compose([RandomAudioTransform(
            random=False),
                                                   transforms.ToTensor(),
                                                   transform_norm])

    def classify(self, image_file_name, checkpoint_file_name,
                 category_names_file_name):
        """classify an image"""
        self.load_checkpoint(checkpoint_file_name)
        self.load_image(image_file_name)
        if category_names_file_name is not None:
            self.load_category_names(category_names_file_name)
        probs, categories = self.infer(self.top_k)
        self.output(probs, categories)

    def infer(self, top_k):
        """Infer the classes"""
        self.model.to(self.device)
        self.model.eval()
        inputs = torch.stack((self.image,))
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        outputs = outputs.to("cpu")
        probs, indices = outputs.topk(top_k)
        probs = probs.exp()
        probs = probs.tolist()[0]
        indices = indices.tolist()[0]
        categories = [self.idx_to_class[index] for index in indices]
        return probs, categories

    def load_category_names(self, category_names_file_name):
        """Load category_names file"""
        with open(category_names_file_name, 'r') as file:
            self.category_names = json.load(file)

    def load_checkpoint(self, checkpoint_file_name):
        """Load checkpoint from file"""
        checkpoint = torch.load(checkpoint_file_name, map_location={'cuda:0': 'cpu'})
        self.model = checkpoint['model']
        self.criterion = nn.NLLLoss()
        class_to_idx = self.model.class_to_idx
        self.idx_to_class = {value : key
                             for key, value in class_to_idx.items()}

    def load_image(self, image_file_name):
        """Load image from file"""
        self.image = Image.open(image_file_name).convert('RGB')
        self.image = self.normalize_image(self.image)

    def normalize_image(self, image):
        """Normalize image"""
        return self.test_transforms(image)

    def output(self, probs, categories):
        """Output category names and propabilities"""
        if self.category_names is not None:
            categories = [self.category_names[category]
                          for category in categories]
            category_title = 'Category Name'
        else:
            category_title = 'Category'
        max_len = max([len(category) for category in categories])
        max_len = max(max_len, len(category_title))
        print('{:>{}} | {}'.format(category_title, max_len, 'Propability'))
        print('{:>{}}-+-{}'.format('-' * max_len, max_len, '-----------'))
        for i in range(len(probs)):
            print('{:>{}} | {:.4f}'.format( categories[i], max_len, probs[i]))

    def set_device(self, device_name):
        """Set the cuda device"""
        self.device = torch.device(device_name)

    def set_top_k(self, top_k):
        """Set number of categories to output"""
        self.top_k = top_k

def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='Classify an image.')
    parser.add_argument('image', help='path to image')
    parser.add_argument('checkpoint', nargs='?', default='checkpoint.pt',
                        help='path to trained model, '
                        'default `checkpoint.tar\'')
    parser.add_argument('-c', '--cpu', action='store_true',
                        help='use CPU for inference')
    parser.add_argument('-g', '--gpu', action='store_true',
                        help='use GPU for inference')
    parser.add_argument('-t', '--top_k', default='5', type=int,
                        help='predict TOP_K best matching categories')
    parser.add_argument('-n', '--category_names',
                        help='json file mapping categories to real name')
    args = parser.parse_args()

    predict = Predict()
    if args.gpu:
        predict.set_device('cuda')
    elif args.cpu:
        predict.set_device('cpu')
    predict.set_top_k(args.top_k)
    predict.classify(args.image, args.checkpoint, args.category_names)

if __name__ == "__main__":
    main()
