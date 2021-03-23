import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pytorch_lightning
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ECGDatasetOneSize(Dataset):
    SIGNAL_PATH_COLUMN = 'PathToData'
    CLASS_COLUMN = 'Disease'
    LENGTH_COLUMN = 'Length'

    def __init__(self, root_dir, df, transform=None, augmentation=None, config=None):
        """

        :param root_dir:
        :param df:
        :param transform:
        :param augmentation:
        :param config:
        """
        if config is None:
            config = {}
        self.root_dir = root_dir
        self.df = df
        self.__create_labels()
        self.transform = transform
        self.augmentation = augmentation
        self.config = config

    def __create_labels(self):
        self.labels = self.df[self.CLASS_COLUMN].apply(self.__label_mapper)

    @staticmethod
    def __label_mapper(label):
        """
        Encode string label to the multi label vector
        Corresponding indexes of the disease:
        0: AF - Atrial fibrillation
        1: I-AVB - First-degree atrioventricular block
        2: LBBB - Left bundle branch block
        3: PAC - Premature atrial complex
        4: PVC - Premature ventricular complex
        5: RBBB - Right bundle branch block
        6: STD - ST-segment depression
        7: STE - ST-segment elevation

        :param label: string with names of the diseases
        :return: encoded multi label y
        """
        mapper = {'AF': 0, 'I-AVB': 1, 'LBBB': 2, 'PAC': 3, 'PVC': 4, 'RBBB': 5, 'STD': 6, 'STE': 7}

        if 'Normal' in label:
            return [0] * len(mapper)

        labels = label.split(',')
        y = [0] * len(mapper)

        for i in range(len(labels)):
            y[mapper[labels[i]]] = 1

        return y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]

        path_to_file = os.path.join(self.root_dir, info[self.SIGNAL_PATH_COLUMN])

        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        y = y.squeeze()

        x = loadmat(path_to_file)
        x = np.asarray(x['val'], dtype=np.float64)

        if self.augmentation:
            x = self.augmentation(x)

        if self.transform:
            x = self.transform(x)

        x = x.view(-1)

        return x.float(), y


class ECGDataset(ECGDatasetOneSize):

    def __init__(self, root_dir, df, size_of_sample, transform=None, augmentation=None, config=None, flatten=False):
        super().__init__(root_dir, df, transform, augmentation, config)
        self.sample_size = size_of_sample
        self.flatten = flatten

    def __getitem__(self, idx):
        info = self.df.iloc[idx]

        path_to_file = os.path.join(self.root_dir, info[self.SIGNAL_PATH_COLUMN])

        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        y = y.squeeze()

        x = np.zeros((12, self.sample_size))
        x_data = loadmat(path_to_file)
        x_data = np.asarray(x_data['val'], dtype=np.float64)

        x[:, :len(x_data[1])] = x_data

        if self.augmentation:
            x = self.augmentation(x)

        if self.transform:
            x = self.transform(x)

        if self.flatten:
            x = x.view(-1)
        else:
            x = torch.squeeze(x)
        return x.float(), y


if __name__ == '__main__':
    path_to_csv = '/datasets/ecg/first_data/labels.csv'
    root_dir = '/datasets/ecg/first_data'
    df = pd.read_csv(path_to_csv)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ECGDataset(root_dir, df, 72000, transform=transform)

    # print("Len of dataet: ", len(dataset))
    for x, y in dataset:
        # print("y: ", y)
        print(x.shape)
        break
