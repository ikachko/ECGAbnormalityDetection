import importlib
import inspect

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch import sigmoid
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

from dataset_samplers.batch_samplers import BalancedBatchSampler
from models import linear_baseline
from models import resnet, fcn, inception
from datasets.entire_ecg_datasets import ECGDataset
from metrics import compute_auc, compute_beta_score, probs_to_hard_predictions


class BaseModel(pl.LightningModule):
    # TODO add test set evaluation
    def __init__(self, hparams, model_params):
        super(BaseModel, self).__init__()
        self.hparams = hparams

        self.model = None
        self.__load_loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        logs_dict = {"Train Loss": loss}
        return {"loss": loss, "log": logs_dict}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        output = self(x)

        prefix = "val" if dataloader_idx else "train"

        loss = self.loss(output, y)

        auroc, auprc = compute_auc(y.cpu().detach().numpy(), sigmoid(output.cpu()).detach().numpy(),
                                   self.hparams.num_classes)

        hard_prediction = probs_to_hard_predictions(output.cpu().detach().numpy(), self.hparams.num_classes)
        accuracy, f_measure, f_beta, g_beta = compute_beta_score(y.cpu().detach().numpy(), hard_prediction, 1, self.hparams.num_classes)

        return {"{}_loss".format(prefix): loss,
                "{}_auroc".format(prefix): torch.tensor(auroc),
                "{}_auprc".format(prefix): torch.tensor(auprc),
                "{}_accuracy".format(prefix): torch.tensor(accuracy),
                "{}_F_measure".format(prefix): torch.tensor(f_measure),
                "{}_F_beta".format(prefix): torch.tensor(f_beta),
                "{}_G_beta".format(prefix): torch.tensor(g_beta),
                "output": output.detach(),
                "y": y.detach()
                }

    def validation_epoch_end(self, outputs):
        # TODO add confusion matrix
        def get_average_metric_value(outputs, dataloaders_names, metric_name):
            metric_values = {}
            for i in range(len(dataloaders_names)):
                average_metric = torch.stack([x["{}_{}".format(dataloaders_names[i], metric_name)] for x in outputs[i]]).mean()
                metric_values["{}_{}".format(dataloaders_names[i], metric_name)] = average_metric
            return metric_values

        losses_logs = get_average_metric_value(outputs, ["train", "val"], "loss")

        auroc_logs = get_average_metric_value(outputs, ["train", "val"], "auroc")
        auprc_logs = get_average_metric_value(outputs, ["train", "val"], "auprc")

        accuracy_logs = get_average_metric_value(outputs, ["train", "val"], "accuracy")
        f_measure_logs = get_average_metric_value(outputs, ["train", "val"], "F_measure")
        f_beta_logs = get_average_metric_value(outputs, ["train", "val"], "F_beta")
        g_beta_logs = get_average_metric_value(outputs, ["train", "val"], "G_beta")

        self.logger.experiment.add_scalars("Losses", losses_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("AUROC", auroc_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("AUPRC", auprc_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("ACCURACY", accuracy_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("F_MEASURE", f_measure_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("F_BETA", f_beta_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("G_BETA", g_beta_logs, global_step=self.current_epoch)

        return {"progress_bar": {**losses_logs, **auroc_logs, **auprc_logs}}

    def configure_optimizers(self):
        mapping = self.__module_mapping('torch.optim')
        return mapping[self.hparams.optimizer](self.model.parameters(),self.hparams.learning_rate)

    def prepare_data(self):
        # create transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # TODO Add cross validation
        df = pd.read_csv(os.path.join(self.hparams.experiment_path, 'labels.csv'))

        train_df, val_df = self.__train_val_split(df)
        train_df.reset_index(inplace=True)
        val_df.reset_index(inplace=True)
        size_of_the_sample = self.hparams.sample_length

        self.train_dataset = ECGDataset(self.hparams.experiment_path, train_df[:1], size_of_the_sample, transform=transform)
        self.eval_train_dataset = ECGDataset(self.hparams.experiment_path, train_df, size_of_the_sample, transform=transform)
        self.val_dataset = ECGDataset(self.hparams.experiment_path, val_df, size_of_the_sample, transform=transform)

    def train_dataloader(self):
        if self.hparams.sampler:
            mapping = self.__module_mapping('dataset_samplers.batch_samplers')
            sampler = mapping[self.hparams.sampler](self.train_dataset)
            return DataLoader(self.train_dataset, sampler=sampler, batch_size=self.hparams.batch_size)
        else:
            return DataLoader(self.train_dataset, num_workers=15, shuffle=True, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return [
                DataLoader(self.eval_train_dataset, num_workers=15, batch_size=self.hparams.batch_size),
                DataLoader(self.val_dataset, num_workers=15, batch_size=self.hparams.batch_size)
               ]

    def __load_loss(self):
        mapping = self.__module_mapping('torch.nn')

        self.loss = mapping[self.hparams.loss]()

    def __train_val_split(self, df):
        train_data, test_data = train_test_split(df, test_size=self.hparams.validation_size,
                                                 random_state=self.hparams.random_state, stratify=df['Disease'])
        return train_data, test_data

    @staticmethod
    def __module_mapping(module_name):
        mapping = {}
        for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
            mapping[name] = obj
        return mapping


