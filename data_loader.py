from collections import Counter

import math
import random
from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import dataset, Dataset
import matplotlib.pyplot as plt

from biotools.models.typings import TaskType

SEED = 1
random.seed(SEED)

__all__ = ['MyDataLoader', 'MyDataset']


@dataclass
class MyDataset:
    train_data: Dataset
    valid_data: Dataset


@dataclass
class MyDataLoader:
    task_type: TaskType
    trait_file: str
    genotype_file: str
    input_dim: int
    k_fold: int = 10
    device: torch.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    return_dataset: bool = True
    is_2d: bool = False
    is_cnn: bool = False
    output_dim: int = 1

    def kfold(self) -> KFold:
        return KFold(n_splits=self.k_fold, shuffle=True, random_state=100)  # have processed before

    def get_size(self):
        # math.floor(math.sqrt(self.input_dim)) 之前是floor
        return math.ceil(math.sqrt(self.input_dim))

    def get_weight(self):
        df = pd.read_csv(self.trait_file, index_col=0)
        y = np.array(df.iloc[:, 0].values.tolist())
        count = Counter(y)
        result = [count[i] for i in range(self.output_dim)]
        return 1. / torch.tensor(result, dtype=torch.float).to(self.device)

    def get_x(self, index, genotype):
        if self.is_2d:
            size = self.get_size()
            x = []
            for i in index:
                x_i = np.array(genotype[i])
                if self.is_cnn:
                    x_i.resize((size, size, 1), refcheck=False)
                else:
                    x_i.resize((size, size), refcheck=False)
                x.append(x_i)
            return np.array(x)
        else:
            return np.array([genotype[i] for i in index])

    def get_train_valid_data(self):
        df = pd.read_csv(self.trait_file, index_col=0)
        genotype_df = pd.read_csv(self.genotype_file, index_col=0).T
        genotype = genotype_df.to_dict('list')
        train_df = df[df['group'] == 'training']
        valid_df = df[df['group'] == 'valid']
        if len(valid_df) == 0:
            valid_df = train_df.head(10)

        if self.task_type == TaskType.CLASSIFICATION:
            train_y = np.array(train_df.iloc[:, 0].values.tolist()).astype(np.int64)
            valid_y = np.array(valid_df.iloc[:, 0].values.tolist()).astype(np.int64)
        elif self.task_type == TaskType.REGRESSION:
            train_y = np.array(train_df.iloc[:, 0].values.tolist())
            valid_y = np.array(valid_df.iloc[:, 0].values.tolist())
        else:
            raise ValueError(f'Task type {self.task_type} not supported')

        train_x = self.get_x(train_df.index.values.tolist(), genotype)
        valid_x = self.get_x(valid_df.index.values.tolist(), genotype)

        if self.return_dataset:
            # x
            if self.is_cnn:
                train_x = torch.FloatTensor(train_x).to(device=self.device).permute(0, 3, 1, 2)
                valid_x = torch.FloatTensor(valid_x).to(device=self.device).permute(0, 3, 1, 2)
            else:
                train_x = torch.FloatTensor(train_x).to(device=self.device)
                valid_x = torch.FloatTensor(valid_x).to(device=self.device)
            # y
            if self.task_type == TaskType.REGRESSION:
                train_y = torch.FloatTensor(train_y).to(self.device).reshape(len(train_y), -1)
                valid_y = torch.FloatTensor(valid_y).to(self.device).reshape(len(valid_y), -1)
            elif self.task_type == TaskType.CLASSIFICATION:
                train_y = torch.tensor(train_y).to(self.device)
                valid_y = torch.tensor(valid_y).to(self.device)
            else:
                raise ValueError(f'Task type {self.task_type} not supported')

            train_dataset = dataset.TensorDataset(train_x, train_y)
            valid_dataset = dataset.TensorDataset(valid_x, valid_y)
            return MyDataset(train_dataset, valid_dataset)
        else:
            return train_x, train_y, valid_x, valid_y
