import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pytorch_lightning
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets.segmentation import sample_batch
from torch.utils.data._utils.collate import default_collate_err_msg_format



def custom_collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        return torch.cat(batch, dim=0)
    elif isinstance(elem, tuple):
        return elem_type(custom_collate_fn(samples) for samples in zip(*batch))
    raise TypeError(default_collate_err_msg_format.format(elem_type))

class TF_Dataset(Dataset):
    def __init__(self, root_dir, ref_file_name, filenames):
        self.root_dir = root_dir
        self.ref_file_name = ref_file_name
        self.file_names = filenames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        annotation_file_name = self.file_names[idx]

        segments, labels, lengths = sample_batch(annotation_file_name, self.ref_file_name, mode='training')

        labels = self.one_hot_encode(labels)

        return torch.tensor(segments).type(torch.float32), torch.tensor(labels).type(torch.float32), torch.tensor(lengths).type(torch.float32)

    @staticmethod
    def one_hot_encode(y):
        encoded_y = []
        for segment in y:
            encoded_segment_label = [0]*9
            encoded_segment_label[segment] = 1
            encoded_y.append(encoded_segment_label)
        return np.array(encoded_y)

if __name__ == "__main__":
    import glob

    def test_val_split_v2(data_path, train_percentage=90):
        file_list = np.array(glob.glob(os.path.join(data_path, '*.mat')))
        no_of_files = len(file_list)
        no_of_train = np.ceil(len(file_list) * train_percentage / 100).astype(np.int32)
        no_of_val = len(file_list) - no_of_train  # not used
        np.random.seed(20)
        index = np.random.permutation(no_of_files)
        train_file_list = list(file_list[index[:no_of_train]])
        val_file_list = list(file_list[index[no_of_train:]])
        #
        return (train_file_list, val_file_list)

    data_root_dir = '/datasets/ecg/tf_data/data/'
    train_file_list, val_file_list = test_val_split_v2(data_root_dir, train_percentage=90)

    ref_file = os.path.join(data_root_dir, 'REFERENCE.csv')
    ds = TF_Dataset(data_root_dir, ref_file, train_file_list)

    dl = DataLoader(ds, collate_fn=collate_fn, batch_size=10)

    for x, y, length in dl:
        print(x.shape)
        print(y.shape)
        break