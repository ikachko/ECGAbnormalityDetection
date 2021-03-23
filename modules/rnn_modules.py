import glob
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from metrics import compute_auc, probs_to_hard_predictions, compute_beta_score
from modules.base import BaseModel
from models.rnn_models import BidirectionalLSTM
from datasets.extracted_features_dataset import TF_Dataset, custom_collate_fn

class BidirectionalLSTMModel(BaseModel):

    def __init__(self, hparams, model_params):
        super().__init__(hparams, model_params)

        self.model = BidirectionalLSTM(
                        input_size=model_params['input_size'],
                        output_size=model_params['output_size'],
                        hidden_dim=model_params['hidden_dim'],
                        n_layers=model_params['n_layers'],
                        drop_prob=model_params['drop_prob'],
                        )

    def training_step(self, batch, batch_nb):
        x, y, lengths = batch
        # x = torch.squeeze(x, 0).type(torch.float32)
        # y = torch.squeeze(y, 0).type(torch.float32)
        output, _ = self(x)

        loss = self.loss(output, y)
        logs_dict = {"Train_Loss": loss}
        return {"loss": loss, "log": logs_dict}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, lengths = batch
        # x = torch.squeeze(x, 0).type(torch.float32)
        # y = torch.squeeze(y, 0).type(torch.float32)
        output, _ = self(x)
        prefix = "val" if dataloader_idx else "train"

        loss = self.loss(output, y)

        auroc, auprc = compute_auc(y.cpu().detach().numpy(), torch.sigmoid(output.cpu()).detach().numpy(),
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

    def prepare_data(self):

        data_root_dir = self.hparams.experiment_path

        train_df, val_df = self.__test_val_split_v2(os.path.join(data_root_dir, 'data'))
        ref_file = os.path.join(data_root_dir, 'REFERENCE.csv')

        self.train_dataset = TF_Dataset(data_root_dir, ref_file, train_df)
        self.eval_train_dataset = TF_Dataset(data_root_dir, ref_file, train_df)
        self.val_dataset = TF_Dataset(data_root_dir, ref_file, val_df)

    def train_dataloader(self):
        if self.hparams.sampler:
            mapping = self.__module_mapping('dataset_samplers.batch_samplers')
            sampler = mapping[self.hparams.sampler](self.train_dataset)
            return DataLoader(self.train_dataset, sampler=sampler, batch_size=self.hparams.batch_size)
        else:
            return DataLoader(self.train_dataset, num_workers=15, shuffle=True, batch_size=self.hparams.batch_size, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return [
                DataLoader(self.eval_train_dataset, num_workers=15, batch_size=self.hparams.batch_size, collate_fn=custom_collate_fn),
                DataLoader(self.val_dataset, num_workers=15, batch_size=self.hparams.batch_size, collate_fn=custom_collate_fn)
               ]
    @staticmethod
    def __test_val_split_v2(data_path, train_percentage=90):
        file_list = np.array(glob.glob(os.path.join(data_path, '*.mat')))
        no_of_files = len(file_list)
        no_of_train = np.ceil(len(file_list) * train_percentage / 100).astype(np.int32)
        no_of_val = len(file_list) - no_of_train  # not used
        np.random.seed(20)
        index = np.random.permutation(no_of_files)
        train_file_list = list(file_list[index[:no_of_train]])
        val_file_list = list(file_list[index[no_of_train:]])
        #
        return train_file_list, val_file_list
