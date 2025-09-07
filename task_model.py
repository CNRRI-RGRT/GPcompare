import json
from typing import List, Union
from dataclasses import dataclass

from biotools.models.typings import TaskType


__all__ = ['TaskModel']


@dataclass(frozen=True)
class TaskModel:
    name: str
    data_dir: str
    task_type: TaskType
    traits: List[str]
    mode: str
    metrics_names: List[str]
    output_dim: Union[int, list]
    genotypes_dir: str
    input_dim: int
    batch_size: int
    hidden_dim: List[list]

    @staticmethod
    def load(file) -> List["TaskModel"]:
        with open(file, 'r') as f:
            data = json.load(f)
        for name, task in data.items():
            task_type = TaskType.get_type(task['task_type'])
            if task_type == TaskType.REGRESSION:
                metrics = ['rmse', 'pearson', 'mae', 'mse', 'R2']
                mode = 'min'
            else:
                metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
                mode = 'max'

            yield TaskModel(
                name=name,
                data_dir=task['data_dir'],
                task_type=task_type,
                mode=mode,
                metrics_names=metrics,
                output_dim=task['output_dim'],
                genotypes_dir=task['genotype_dir'],
                input_dim=task['input_dim'],
                traits=task['traits'],
                hidden_dim=task['hidden_size'],
                batch_size=task['batch_size'],
            )
