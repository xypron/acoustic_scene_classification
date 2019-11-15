#!/usr/bin/python3
# SPDX-License-Identifier: LGPL-2.0
#
# Heinrich Schuchardt

"""
This module demonstrates the training of a model.
"""

import argparse
from collections import OrderedDict
import math
import multiprocessing
import os
from PIL import Image, ImageDraw, ImageOps
import random
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

class FrequencyMask:
    """Randomly mask frequency band in short time spectrogram

    The spectrogram is transformed randomly in three ways:

    * The spectrogram frequencies are transposed randomly.
    * The speed is randomly changed.
    * A random sampling window in time is chosen.

    max_width        - maximum portion of all frequencies that will be masked
    """

    def __init__(self, max_width = .2):
        """Construct new FrequencyMask
        """

        self.max_width = max_width

    def __call__(self, img):
        """Transform a spectrogram provided as as image

        img    - image to transform
        Return - transformed image
        """
        width, height = img.size
        mask_height = height * self.max_width * random.random()
        ymin = math.floor((height - mask_height) * random.random())
        ymax = math.floor(mask_height + ymin)
        draw = ImageDraw.Draw(img)
        draw.rectangle(((0, ymin), (width - 1, ymax)), fill=(127, 127, 127))
        return img

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

    def __init__(self, size=224, octaves=.5, bins_per_octave=24, dilation=0.0,
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

class Train:
    """Train network"""

    def __init__(self, data_dir, arch, epochs, hidden_units, learning_rate,
                 save_dir):
        """Constructor"""
        self.model_name = arch
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        """Train a model """
        self.create_data_loaders()
        self.load_model()
        self.create_classifier()
        self.optimize()
        self.validate()
        self.save_checkpoint()
        self.free_gpu()

    def create_classifier(self):
        """Create the classifier and assign it to the model"""
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.get_in_features(), self.hidden_units)),
            ('relu1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(.5)),
            ('fc2', nn.Linear(self.hidden_units, self.hidden_units)),
            ('relu2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(.5)),
            ('fc3', nn.Linear(self.hidden_units, len(self.class_to_idx))),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        self.model.classifier = classifier

    def create_data_loaders(self):
        """Create the data loaders"""
        transform_norm = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([RandomAudioTransform(
            random=True),
                                               #FrequencyMask(max_width=.3),
                                               transforms.ToTensor(),
                                               transform_norm])

        test_transforms = transforms.Compose([RandomAudioTransform(
            random=False),
                                              transforms.ToTensor(),
                                              transform_norm])

        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')
        valid_dir = os.path.join(self.data_dir, 'valid')

        train_data = datasets.ImageFolder(
            train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

        self.class_to_idx = test_data.class_to_idx

        self.trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=32, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(
            valid_data, batch_size=32, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(
            test_data, batch_size=32, shuffle=True)

    def do_deep_learning(self, model, criterion, optimizer, print_every=40):
        """Execute training steps"""

        model.to(self.device)
        model.train()

        for epoch in range(self.epochs):
            running_loss = 0
            steps = 0
            for inputs, labels in self.trainloader:
                steps += 1

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                print('.', end='', flush=True)
                if steps % print_every == 0:
                    print("Epoch: {}/{}, ".format(epoch + 1, self.epochs),
                          "Loss: {:.4f}".format(running_loss/print_every))
                    running_loss = 0
            while steps % print_every != 0:
                print(' ', end='')
                steps += 1
            print("Epoch {} completed".format(epoch + 1))
            self.check_accuracy(model, self.testloader)
        print()

    def check_accuracy(self, model, testloader, print_every=40):
        """Check the accuracy of the model"""
        correct = 0
        total = 0
        steps = 0

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for inputs, labels in testloader:
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                print('-', end='', flush=True)
                if steps % print_every == 0:
                    print()
        while steps % print_every != 0:
            print(' ', end='')
            steps += 1
        print('Accuracy: {:.4f} %'.format(100 * correct / total))

    def load_model(self):
        """Load model"""
        method_to_call = getattr(models, self.model_name)

        self.model = method_to_call(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

    def free_gpu(self):
        """Free CUDA resources"""
        try:
            self.model.to('cpu')
            del self.model
        except Exception:
            pass
        torch.cuda.empty_cache()

    def get_in_features(self):
        """Get the number of in_features of the classifier"""
        self.in_features = 0
        for module in self.model.classifier.modules():
            try:
                self.in_features = module.in_features
                break
            except AttributeError:
                pass
        return self.in_features

    def optimize(self):
        """Execute the optimization"""
        self.optimizer = optim.Adam(self.model.classifier.parameters(),
                                    lr=self.learning_rate)
        criterion = nn.NLLLoss()
        self.do_deep_learning(self.model, criterion, self.optimizer)

    def save_checkpoint(self):
        """Save a checkpoint"""
        path = os.path.join(self.save_dir, 'checkpoint.pt')
        self.model.class_to_idx = self.class_to_idx
        torch.save({'epoch': self.epochs,
                    'model': self.model,
                    'optimizer': self.optimizer}, path)

    def validate(self):
        """Check the accuracy with the validation data set"""
        print("Validation")
        self.check_accuracy(self.model, self.validloader)
        print('model = {}, hidden units = {}, learning rate = {}'
              .format(self.model_name, self.hidden_units, self.learning_rate))

    def set_device(self, device_name):
        """Set the cuda device"""
        self.device = torch.device(device_name)

def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('data_dir', nargs='?', default='data',
                        help='directory with training data')
    parser.add_argument('-a', '--arch', default='vgg16',
                        help='torchvision.model, default vgg16')
    parser.add_argument('-e', '--epochs', default='4',
                        help='number of epochs, default 4')
    parser.add_argument('-u', '--hidden_units', default='512',
                        help='hidden units per layer, default 512')
    parser.add_argument('-l', '--learning_rate', default='.001',
                        help='learning_rate, default .001')
    parser.add_argument('-s', '--save_dir', default='.',
                        help='directory in which the checkpoint will be saved')
    parser.add_argument('-c', '--cpu', action='store_true',
                        help='use CPU for inference')
    parser.add_argument('-g', '--gpu', action='store_true',
                        help='use GPU for inference')
    args = parser.parse_args()

    # Use all cores (including simultaneous multithreading)
    torch.set_num_threads(multiprocessing.cpu_count())

    train = Train(args.data_dir, args.arch, int(args.epochs),
                  int(args.hidden_units), float(args.learning_rate),
                  args.save_dir)
    if args.gpu:
        train.set_device('cuda')
    elif args.cpu:
        train.set_device('cpu')
    train.train()

if __name__ == "__main__":
    main()
