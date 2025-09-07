from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data_loader import MyDataset
from biotools.utils.evaluator import evaluator
from biotools.models.typings import TaskType

from task_model import TaskModel

__all__ = ['Trainer']


def plot(train_plot_loss: list, val_plot_loss: list):
    plt.plot(train_plot_loss[2:], label='train', color='blue')
    plt.plot(val_plot_loss[2:], label='val', color='red')
    plt.legend()
    plt.show()


@dataclass
class Trainer:
    data: MyDataset
    task: TaskModel
    device: torch.device
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.001
    loss_fn: Callable = nn.MSELoss()

    @staticmethod
    def get_best_params(model, best_state_dict: dict, best_metric: float, metric_score: float, task_type) -> (
            Tuple)[dict, float]:
        if task_type == TaskType.REGRESSION:
            if metric_score < best_metric:
                best_metric = metric_score
                best_state_dict = deepcopy(model.state_dict())
        elif task_type == TaskType.CLASSIFICATION:
            if metric_score > best_metric:
                best_metric = metric_score
                best_state_dict = deepcopy(model.state_dict())
        else:
            raise ValueError(f'Task type {task_type} not supported')
        return best_state_dict, best_metric

    def train(self, model: nn.Module):
        train_data = DataLoader(self.data.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)

        if self.task.task_type == TaskType.REGRESSION:
            best_metrics = float('+inf')
            mode = 'min'
        elif self.task.task_type == TaskType.CLASSIFICATION:
            best_metrics = float('-inf')
            mode = 'max'
        else:
            raise ValueError(f'Task type {self.task.task_type} not supported')

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode)
        train_plot_loss = []
        valid_plot_loss = []
        best_state_dict = {}
        log = []
        for epoch in range(self.epochs):
            torch.cuda.empty_cache()
            model.train()
            train_y_true = []
            train_y_pred = []
            loss_list = []
            for x, y in train_data:
                optimizer.zero_grad()
                result = model(x)
                total_loss = self.loss_fn(result, y)
                total_loss.backward()
                optimizer.step()
                if self.task.task_type == TaskType.CLASSIFICATION:
                    _, result = torch.max(result, 1)
                train_y_pred.extend(result.data.tolist())
                train_y_true.extend(y.data.tolist())
                loss_list.append(total_loss.item())
            val_y_true, val_pred, val_loss = self.validate(model, self.data.valid_data)
            scheduler.step(val_loss)
            train_loss = np.mean(loss_list)
            train_plot_loss.append(train_loss)
            valid_plot_loss.append(val_loss)
            metrics1 = self.task.metrics_names[0]
            metrics2 = self.task.metrics_names[1]
            valid_best_metrics = evaluator(val_y_true, val_pred, metrics1)

            # print
            # if epoch % 10 == 0:
            log.append(f'Training Epoch: {epoch}, Loss: {train_loss:.4f}, '
                       f'{metrics1}: {evaluator(train_y_true, train_y_pred, metrics1)}, '
                       f'{metrics2}: {evaluator(train_y_true, train_y_pred, metrics2)}')

            log.append(f'Valid Epoch: {epoch}, Loss: {val_loss:.4f}, '
                       f'{metrics1}: {valid_best_metrics}, '
                       f'{metrics2}: {evaluator(val_y_true, val_pred, metrics2)}')

            best_state_dict, best_metrics = (
                self.get_best_params(model, best_state_dict, best_metrics, valid_best_metrics,
                                     self.task.task_type))

        # plot(train_plot_loss, valid_plot_loss)
        if best_state_dict:
            model.load_state_dict(best_state_dict)

        return model, best_metrics, log

    @torch.no_grad()
    def validate(self, val_model: nn.Module, data):
        val_model.eval()
        val_data = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        val_y_true = []
        val_pred = []
        loss_list = []
        for x, y in val_data:
            result = val_model(x)
            total_loss = self.loss_fn(result, y)
            if self.task.task_type == TaskType.CLASSIFICATION:
                _, result = torch.max(result, 1)
            val_y_true.extend(y.data.tolist())
            val_pred.extend(result.data.tolist())
            loss_list.append(total_loss.item())
        return val_y_true, val_pred, np.mean(loss_list)
