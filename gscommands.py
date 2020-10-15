import torch as th
import torchaudio
from torch import utils
from torchaudio import transforms
from torchaudio.datasets.utils import walk_files

import os
import shutil
import argparse
import pickle

from typing import Tuple
from nnchassis import PrintUtils as P

# Define globals
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
TRAIN_FOLDER = "train"
VALID_FOLDER = "valid"
TEST_FOLDER = "test"
SAMPLE_LENGTH = 16000
CLASSES_FILE = "classes.pickle"

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
    classes = {}
    class_idx = 0
    for _,item in enumerate(directory_contents):
        if os.path.isdir(gcommands_folder+'/'+item):
            if not item == EXCEPT_FOLDER:
              classes[item] = class_idx
              class_idx += 1
    # classes.pop(EXCEPT_FOLDER,None)

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

    with open(CLASSES_FILE, 'wb') as handle:
        pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    move_files(gcommands_folder, test_folder, test_path)
    move_files(gcommands_folder, valid_folder, validation_path)
    create_train_folder(gcommands_folder, train_folder, test_folder)
    shutil.copy(CLASSES_FILE, test_folder)
    shutil.copy(CLASSES_FILE, train_folder)
    shutil.copy(CLASSES_FILE, valid_folder)

class GoogleCommands(utils.data.Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label, speaker_id, utterance_number
    """

    def __init__(self,
                 root_path: str,
                 download: bool = False) -> None:

        self._path = os.path.join(root_path)
        classes_file_path = os.path.join(root_path, CLASSES_FILE)

        walker = walk_files(self._path, suffix=".wav", prefix=True)
        walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
        self._walker = list(walker)
        with open(classes_file_path, 'rb') as handle:
            self.classes = pickle.load(handle)

    def __getitem__(self, n: int) -> Tuple[th.Tensor, int, str, str, int]:
        fileid = self._walker[n]
        return self.load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)

    def load_speechcommands_item(self, filepath: str, path: str) -> Tuple[th.Tensor, int, str, str, int]:
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
            waveform = wave
        waveform = th.squeeze(waveform)
        # return waveform, sample_rate, label, speaker_id, utterance_number
        return waveform, self.classes[label]    

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
      download_marker_file = os.path.join(root_path, "downloaded")
      if os.path.isfile(download_marker_file):
        x = input("Database seems downloaded. Wipe and download [y|n]?:")
        download = True if (x == "y" or x == "Y") else False

      if download:
          shutil.rmtree(root_path)
          os.mkdir(root_path)
          # Download the dataset using the default dataset from Pytorch audio
          _ = torchaudio.datasets.SPEECHCOMMANDS(root=root_path,download=True)
          # then separate the test, train and validation sets
          make_dataset(gcommands_folder=src_path, out_path=out_path)
          dld_mark = open(download_marker_file, "w")
          dld_mark.write("True")
          dld_mark.close()

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
                                            shuffle=False)
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
    return gen_dataloaders( root_path=root_path, 
                            download=download, 
                            train_batch_size=train_batch_size,
                            test_batch_size=test_batch_size,
                            val_batch_size=val_batch_size)
