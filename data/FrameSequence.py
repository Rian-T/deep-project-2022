from torch.utils.data import Dataset
from PIL import Image
import torch
import os 
import re
from torchvision import transforms
import numpy

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

# a dataset containing sequences of 16 images of 84x84 pixels
class FrameSequenceDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, root_dir="datasets/MarioKartFrameSequence16", train=True, transform=transforms.ToTensor()):
        
        if train:
            circuits = ["BowserCastle_M", "ChocoIsland_M", "KoopaBeach_M"]
        else:
            circuits = ["MarioCircuit_M"]
        
        self.labels = []
        self.images = []

        self.transforms = transform
        self.nb_images = None

        for c in circuits:
            print("Loading " + c)
            dataset_dir = root_dir + "/" + c
            for x in os.walk(dataset_dir):
                for y in os.walk(x[0]):
                    if len(y[0].split("\\")) > 1:
                        action = y[0].split("\\")[1]
                        images_path = y[2]
                        images_path.sort(key=alphanum_key)
                        if len(images_path) >= 8:
                            self.nb_images = len(images_path)
                            self.labels.append(action)
                            # Concatanate all images into a single tensor
                            images_numpy = numpy.asarray([numpy.asarray(Image.open(y[0] + '/' + images_path[i])) for i in range(self.nb_images)])
                            self.images.append(images_numpy)

        print("Loaded " + str(len(self.labels)) + " samples")

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        images_numpy = self.images[index]
        images = self.transforms((images_numpy)).reshape(self.nb_images, 84, 84)
        label = torch.tensor(int(self.labels[index]))
        return images, label