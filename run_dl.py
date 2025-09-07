import os.path
from itertools import product

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from biotools.models import creat_mlp_layer, resnet18, resnet34, RNN, LSTM, CNNModel,\
    xgboost_ml, random_forest, TaskType, lightgbm, CNNModel
from biotools.utils import evaluator


from data_loader import MyDataLoader
from task_model import TaskModel
from training import Trainer


models = {
    'DNN': {"func": creat_mlp_layer, "2d": False, 'return_dataset': True, 'cnn': False,
            'lrs': [1e-3, 1e-4], "dropouts": [0.4, 0.5]},
    'LSTM': {"func": LSTM, "2d": True, 'return_dataset': True, 'cnn': False,
             'lrs': [1e-3, 1e-4], "dropouts": [0.4]},
    'RNN': {"func": RNN, "2d": True, 'return_dataset': True, 'cnn': False,
            'lrs': [1e-3, 1e-4], "dropouts": [0.4]},
    'ResNet18': {"func": resnet18, "2d": True, 'return_dataset': True, 'cnn': True,
                 'lrs': [1e-3, 1e-4], "dropouts": [0.1, 0.2]},
    'ResNet34': {"func": resnet34, "2d": True, 'return_dataset': True, 'cnn': True,
                 'lrs': [1e-3, 1e-4], "dropouts": [0.1, 0.2]},
}

k_fold = 10
input_dir = '/home/input_data'
output_dir = '/home/output_data'
input_json = '/home/input_data.json'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 每一个task
tasks = list(TaskModel.load(input_json))
for task in tasks:
    task_result_ave = [
        ['ModelName', 'Trait', 'Task', 'TaskType', 'rmse', 'pearson', 'mae', 'mse', 'r2']]  # 每个模型总共的task总结
    temp_valid_score_list = []
    for model_name, detail in models.items():
        print(model_name)
        # 每一个性状
        for trait_id, trait in enumerate(task.traits):
            os.makedirs(os.path.join(output_dir, task.name), exist_ok=True)
            print(f"{task.name}: {trait}: {model_name} Training")
            genotype_data = os.path.join(input_dir, task.data_dir, task.genotypes_dir)

            # 10倍交叉验证
            output_dim = task.output_dim[trait_id] if isinstance(task.output_dim, list) else task.output_dim
            valid_score_list = []
            train_score_list = []
            for idx in range(k_fold):
                logs = []
                print(f'data cross validation: {idx}')
                trait_file = os.path.join(input_dir, task.data_dir, f'{trait}_processed_cross{idx}.csv')
                data_load = MyDataLoader(
                    trait_file=trait_file, genotype_file=genotype_data, k_fold=k_fold, is_2d=detail['2d'],
                    task_type=task.task_type, output_dim=output_dim,
                    return_dataset=detail['return_dataset'], input_dim=task.input_dim, is_cnn=detail['cnn'],
                    device=device)
                data = data_load.get_train_valid_data()
                hidden_dim = [1]
                if 'Net' not in model_name:
                    hidden_dim = task.hidden_dim
                hyper_param = product(models[model_name]['lrs'], models[model_name]['dropouts'], hidden_dim)
                # loss_fn
                if task.task_type == TaskType.REGRESSION:
                    loss_fn = nn.MSELoss()
                    init_best_metrics = float('+inf')  # 相对于每一份训练集来说的
                else:
                    loss_fn = nn.CrossEntropyLoss(weight=data_load.get_weight())
                    init_best_metrics = float('-inf')

                for lr, dropout, hidden_size in hyper_param:
                    print(lr, dropout, hidden_size)
                    if model_name == 'DNN':
                        model = creat_mlp_layer(
                            input_dim=task.input_dim, output_dim=output_dim,
                            hidden_dims=hidden_size, dropout=dropout, task_type=task.task_type).to(device)
                    elif 'Net' in model_name:
                        model = detail['func'](in_channel=1, out_dim=output_dim, dropout=dropout,
                                               task_type=task.task_type).to(device)
                    elif model_name == 'LSTM':
                        model = LSTM(
                            input_size=data_load.get_size(), hidden_size=data_load.get_size(),
                            output_size=output_dim, dropout=dropout, hidden_dims=hidden_size,
                            task_type=task.task_type).to(device)
                    elif model_name == 'RNN':
                        model = RNN(
                            input_size=data_load.get_size(), hidden_size=data_load.get_size(),
                            output_size=output_dim, hidden_dims=hidden_size,
                            dropout=dropout, task_type=task.task_type).to(device)
                    else:
                        raise ValueError(f'Model {model_name} not recognized')
                    epochs = 100
                    if lr < 1e-3:
                        epochs = 150
                    trainer = Trainer(data=data, device=device, epochs=epochs,
                                      batch_size=task.batch_size, lr=lr, task=task, loss_fn=loss_fn)
                    model, best_metrics, log = trainer.train(model)
                    logs.append(f'{lr}, {dropout}, {hidden_size}')
                    logs.extend(log)

                    val_y_true, val_pred, _ = trainer.validate(model, data.valid_data)
                    train_y_true, train_y_pred, _ = trainer.validate(model, data.train_data)
                    val_metrics_scores = []
                    train_metrics_scores = []
                    for metric in task.metrics_names:
                        score = evaluator(val_y_true, val_pred, metric)
                        val_metrics_scores.append(score)
                        train_metrics_scores.append(evaluator(train_y_true, train_y_pred, metric))

                    if task.task_type == TaskType.REGRESSION:  # rmse
                        if best_metrics < init_best_metrics:
                            val_best_score = val_metrics_scores
                            train_best_score = train_metrics_scores
                            init_best_metrics = best_metrics
                            y_pred_val = val_pred
                            y_true_val = val_y_true
                            y_pred_train = train_y_pred
                            y_true_train = train_y_true
                            best_model = deepcopy(model)
                    elif task.task_type == TaskType.CLASSIFICATION:  # accuracy
                        if best_metrics > init_best_metrics:
                            val_best_score = val_metrics_scores
                            init_best_metrics = best_metrics
                            train_best_score = train_metrics_scores
                            y_pred_val = val_pred
                            y_true_val = val_y_true
                            y_pred_train = train_y_pred
                            y_true_train = train_y_true
                            best_model = deepcopy(model)
                    else:
                        raise ValueError(f'Model {model_name} not recognized')

                with open(os.path.join(output_dir, task.name, f'{model_name}_{task.name}_{trait}_log{idx}.txt'),
                          'w') as f:
                    f.write('\n'.join(logs))

                torch.save(best_model, os.path.join(output_dir, task.name, f'{trait}_{model_name}_{idx}.pt'))

                if task.task_type == TaskType.REGRESSION:
                    y_pred_val = [x[0] for x in y_pred_val]
                    y_true_val = [x[0] for x in y_true_val]
                    y_pred_train = [x[0] for x in y_pred_train]
                    y_true_train = [x[0] for x in y_true_train]

                test_data = pd.DataFrame({'y_pred': y_pred_val, 'y_true': y_true_val})
                train_data = pd.DataFrame({'y_pred': y_pred_train, 'y_true': y_true_train})
                test_data.to_csv(
                    os.path.join(output_dir, task.name, f'{model_name}_{task.name}_{trait}_Pred_Valid_Cross{idx}.csv'),
                    index=False)
                train_data.to_csv(
                    os.path.join(output_dir, task.name, f'{model_name}_{task.name}_{trait}_Pred_Train_Cross{idx}.csv'),
                    index=False)
                for metrics, score in zip(task.metrics_names, val_best_score):
                    print(f'valid {metrics}: {score} ')
                valid_score_list.append(val_best_score)
                train_score_list.append(train_best_score)
                temp_valid_score_list.append(
                    [model_name, trait, task.name, task.task_type.name, 'training'] + train_best_score)
                temp_valid_score_list.append(
                    [model_name, trait, task.name, task.task_type.name, 'valid'] + val_best_score)
                df1 = pd.DataFrame(temp_valid_score_list)
                df1.to_csv(os.path.join(output_dir, f'{task.name}_cross.csv'), index=False)

            task_result_ave.append(
                [model_name, trait, task.name, task.task_type.name, 'training'] + list(
                    np.average(np.array(train_score_list), axis=0)))
            task_result_ave.append(
                [model_name, trait, task.name, task.task_type.name, 'valid'] + list(
                    np.average(np.array(valid_score_list), axis=0)))
            df = pd.DataFrame(task_result_ave)
            df.to_csv(os.path.join(output_dir, f'{task.name}.csv'), index=False)
