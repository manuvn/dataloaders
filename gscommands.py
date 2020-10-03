import torch as th
import torchaudio
from torch import utils
from torchaudio import transforms
from torchaudio.datasets.utils import walk_files

import os
import shutil
import argparse

from typing import Tuple
from nnchassis import PrintUtils as P

# Define globals
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
TRAIN_FOLDER = "train"
VALID_FOLDER = "valid"
TEST_FOLDER = "test"
SAMPLE_LENGTH = 16000
CLASSES = {}

# Some utility function
def move_files(original_folder, data_folder, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_folder = os.path.join(data_folder, vals[0])
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
            shutil.move(os.path.join(original_folder, line[:-1]), os.path.join(data_folder, line[:-1]))


def create_train_folder(original_folder, data_folder, test_folder):
    # list dirs
    dir_names = list()
    for file in os.listdir(test_folder):
        if os.path.isdir(os.path.join(test_folder, file)):
            dir_names.append(file)

    # build train folder
    for file in os.listdir(original_folder):
        if os.path.isdir(os.path.join(test_folder, file)) and file in dir_names:
            shutil.move(os.path.join(original_folder, file), os.path.join(data_folder, file))


def make_dataset(gcommands_folder, out_path):
    validation_path = os.path.join(gcommands_folder, 'validation_list.txt')
    test_path = os.path.join(gcommands_folder, 'testing_list.txt')
    directory_contents = os.listdir(gcommands_folder)
    for idx,item in enumerate(directory_contents):
      if os.path.isdir(gcommands_folder+'/'+item):
        CLASSES[item] = idx
    CLASSES.pop(EXCEPT_FOLDER,None)

    valid_folder = os.path.join(out_path, VALID_FOLDER)
    test_folder = os.path.join(out_path, TEST_FOLDER)
    train_folder = os.path.join(out_path, TRAIN_FOLDER)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(valid_folder):
        os.mkdir(valid_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    move_files(gcommands_folder, test_folder, test_path)
    move_files(gcommands_folder, valid_folder, validation_path)
    create_train_folder(gcommands_folder, train_folder, test_folder)

def load_speechcommands_item(filepath: str, path: str) -> Tuple[th.Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    if (waveform.shape[1] < SAMPLE_LENGTH):
        # pad early with zeros in case the sequence is shorter than 16000 samples
        wave = th.zeros([1,SAMPLE_LENGTH])
        wave[0,-waveform.shape[1]:] = waveform
        waveform = th.squeeze(wave)
    # return waveform, sample_rate, label, speaker_id, utterance_number
    return waveform, CLASSES[label]

class GoogleCommands(utils.data.Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label, speaker_id, utterance_number
    """

    def __init__(self,
                 root_path: str,
                 download: bool = False) -> None:

        self._path = os.path.join(root_path)
        walker = walk_files(self._path, suffix=".wav", prefix=True)
        walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[th.Tensor, int, str, str, int]:
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)

def gen_dataloaders(root_path,
                    train_batch_size=32, 
                    test_batch_size=32, 
                    val_batch_size=32,
                    download=False):

    root_path = os.path.join(root_path)
    src_path = os.path.join(root_path, 'SpeechCommands','speech_commands_v0.02')
    out_path = os.path.join(root_path, "SpeechCommands")

    if not download:
      if not os.path.exists(out_path):
        P.print_message("Looks like dataset folder is missing. You may want check and enable download")
    else:
      # first download the dataset
      if os.path.exists(src_path):
        shutil.rmtree(root_path)
    
      # Download the dataset using the default dataset from Pytorch audio
      _ = torchaudio.datasets.SPEECHCOMMANDS(root=root_path,
                                                        download=True)

      # then separate the test, train and validation sets
      make_dataset(gcommands_folder=src_path, out_path=out_path)

    # Now create thed dataloaders
    train_path = os.path.join(out_path, TRAIN_FOLDER)
    train_dataset = GoogleCommands(root_path=train_path)
    val_path = os.path.join(out_path, VALID_FOLDER)
    val_dataset = GoogleCommands(root_path=val_path)
    test_path = os.path.join(out_path, TEST_FOLDER)
    test_dataset = GoogleCommands(root_path=test_path)

    train_dataloader = utils.data.DataLoader(train_dataset, 
                                            batch_size=train_batch_size,
                                            shuffle=True)
    
    val_dataloader = utils.data.DataLoader(val_dataset, 
                                            batch_size=val_batch_size, 
                                            shuffle=True)

    test_dataloader = utils.data.DataLoader(test_dataset, 
                                            batch_size=test_batch_size, 
                                            shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

def gscommands_gen(root_path=None, 
                    train_batch_size=32, 
                    test_batch_size=32, 
                    val_batch_size=32,
                    download=False):
    """
    Create dataloaders for the Google Speech Commands task. Each item is a tuple of the form:
    waveform, sample_rate, label, speaker_id, utterance_number.
    Return train, validation and test dataloaders
    """
    if root_path is None:
        root_path = os.path.join(os.getcwd(), "dataset")
        if not os.path.exists(root_path):
            os.mkdir(root_path)
    return gen_dataloaders(root_path=root_path, download=download)