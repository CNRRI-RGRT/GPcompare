import os.path

from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
from biotools.models import svm_ml, xgboost_ml, random_forest, lightgbm
from biotools.utils import evaluator

from data_loader import MyDataLoader
from task_model import TaskModel

models = {
    'SVM': {"func": svm_ml, "2d": False, 'return_dataset': False, 'cnn': False},
    'XGBoost': {"func": xgboost_ml, "2d": False, 'return_dataset': False, 'cnn': False},
    'RF': {"func": random_forest, "2d": False, 'return_dataset': False, 'cnn': False},
    'LightGBM': {"func": lightgbm, "2d": False, 'return_dataset': False, 'cnn': False},
}
k_fold = 10

input_dir = '/home/data/processed'
output_dir = '/home/models'

# 每一个模型
#  ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']

os.makedirs(output_dir, exist_ok=True)
tasks = list(TaskModel.load('sample.json'))


def cross_training(idx, trait, task, model_name, detail):
    genotype_data = os.path.join(input_dir, task.data_dir, task.genotypes_dir)
    trait_file = os.path.join(input_dir, task.data_dir, 'sample3', f'{trait}_processed_cross{idx}.csv')
    data_loader = MyDataLoader(
        trait_file=trait_file, genotype_file=genotype_data, k_fold=k_fold, is_2d=detail['2d'],
        task_type=task.task_type,
        return_dataset=detail['return_dataset'], input_dim=task.input_dim, is_cnn=detail['cnn'],
        device='cpu')
    print(f'data cross validation: {idx}')
    train_x, train_y, test_x, test_y = data_loader.get_train_valid_data()
    val_y_pred, train_y_pred, _ = detail['func'](train_x, train_y, test_x, test_y, task_type=task.task_type)
    test_data = pd.DataFrame({'y_pred': val_y_pred, 'y_true': test_y})
    train_data = pd.DataFrame({'y_pred': train_y_pred, 'y_true': train_y})
    os.makedirs(os.path.join(output_dir, task.name), exist_ok=True)
    test_data.to_csv(
        os.path.join(output_dir, task.name, f'{model_name}_{task.name}_{trait}_Pred_Valid_Cross{idx}.csv'),
        index=False)
    train_data.to_csv(
        os.path.join(output_dir, task.name, f'{model_name}_{task.name}_{trait}_Pred_Train_Cross{idx}.csv'),
        index=False)
    val_metrics_scores = []
    train_metrics_scores = []
    for metric in task.metrics_names:
        score = evaluator(test_y, val_y_pred, metric)
        val_metrics_scores.append(score)
        train_metrics_scores.append(evaluator(train_y, train_y_pred, metric))

    return val_metrics_scores, train_metrics_scores


def run():
    # 每一个task
    task_result_raw = [['ModelName', 'Trait', 'Task', 'TaskType', 'rmse', 'pearson', 'mae', 'mse', 'r2']]
    task_result_ave = [['ModelName', 'Trait', 'Task', 'TaskType', 'rmse', 'pearson', 'mae', 'mse', 'r2']]
    for task in tasks:
        for model_name, detail in models.items():
            print(model_name)
            # 每一个性状
            for trait in task.traits:
                pool = Pool(processes=2)
                print(f"{task.name}: {trait}: {model_name} Training")
                train_model = partial(cross_training, trait=trait, task=task, model_name=model_name, detail=detail)
                results = pool.map(train_model, range(k_fold))
                pool.close()
                pool.join()

                # 10倍交叉验证
                val_score_list = []
                train_score_list = []
                for val_metrics_scores, train_metrics_scores in results:
                    val_score_list.append(val_metrics_scores)
                    train_score_list.append(train_metrics_scores)
                    task_result_raw.append(
                        [model_name, trait, task.name, task.task_type.name, 'valid'] + val_metrics_scores)
                    task_result_raw.append(
                        [model_name, trait, task.name, task.task_type.name, 'training'] + train_metrics_scores)

                task_result_ave.append(
                    [model_name, trait, task.name, task.task_type.name, 'valid'] + list(
                        np.average(np.array(val_score_list), axis=0)))
                task_result_ave.append(
                    [model_name, trait, task.name, task.task_type.name, 'training'] + list(
                        np.average(np.array(train_score_list), axis=0)))

                df_raw = pd.DataFrame(task_result_raw)
                df_raw.to_csv(os.path.join(output_dir, f'{task.name}_{model_name}_cross.csv'), index=False)

                df = pd.DataFrame(task_result_ave)
                df.to_csv(os.path.join(output_dir, f'{task.name}_{model_name}_average.csv'), index=False)


if __name__ == '__main__':
    run()
