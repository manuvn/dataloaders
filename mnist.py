# For data loader
import torch as th
import torchvision
from torch import utils
from torchvision import transforms

# Dataset preparation
class Mnist1DTransform(object):
  """
  Converts an MNIST image to a sequence. Ouput shape: 28*28
  """
  def __call__(self, sample):
    return th.reshape(sample, [28*28])

class MnistSeqTransform(object):
  """
  Converts an MNIST image to a sequence. Ouput shape: 28*28,1
  """
  def __call__(self, sample):
    return th.reshape(sample, [28*28,1])

def gen_dataloaders(data_ops, train_batch_size=8, val_batch_size=8, root_path = '.'):
    train_dataset = torchvision.datasets.MNIST(root=root_path, 
                                            train=True, 
                                            download=True,
                                            transform=transforms.Compose(data_ops))

    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = torchvision.datasets.MNIST(root=root_path, 
                                            train=False, 
                                            download=True,
                                            transform=transforms.Compose(data_ops))

    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

    return train_dataloader, val_dataloader

def mnist_gen(train_batch_size=8, val_batch_size=8, flatten=False, sequential=False):
    """
    Flatten = false will reshape images to be a 1D vector. Ex use for MLP.
    """
    data_ops = [transforms.ToTensor()]

    if flatten:
        data_ops =[transforms.ToTensor(), Mnist1DTransform()]

    if sequential:
        data_ops =[transforms.ToTensor(), MnistSeqTransform()]

    return gen_dataloaders(data_ops, train_batch_size, val_batch_size, root_path='./dataset/.')