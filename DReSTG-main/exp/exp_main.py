# DReSTG train, validation and test; compact metrics only.
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DReSTG
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.drestg_identity import (
    cell_identity_from_args, recipe_from_args, semantic_fingerprint,
    screen_semantics, fusion_semantics, relation_settlement,
)

import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {'DReSTG': DReSTG}
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) != 5:
            raise ValueError('DReSTG runtime requires Reference payload; unexpected batch arity=%d' % len(batch))
        return batch

    def _forward_batch(self, batch_x, relation):
        return self.model(batch_x, relation_cache=relation)

    def _semantic_fingerprint(self):
        return semantic_fingerprint()

    def _cell_identity(self):
        return cell_identity_from_args(self.args)

    def _recipe(self):
        return recipe_from_args(self.args)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion_y, mode_flag='vail'):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, relation = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_y = self._forward_batch(batch_x, relation)
                else:
                    output_y = self._forward_batch(batch_x, relation)

                batch_y_y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
                pred_y = output_y.detach().cpu()
                true_y = batch_y_y.detach().cpu()
                loss = criterion_y(pred_y, true_y)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion_y = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, relation = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_y = self._forward_batch(batch_x, relation)
                        batch_y_y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
                        loss = criterion_y(output_y, batch_y_y)
                        train_loss.append(loss.item())
                else:
                    output_y = self._forward_batch(batch_x, relation)
                    batch_y_y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
                    loss = criterion_y(output_y, batch_y_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print('\titers: {0}, epoch: {1} | loss: {2:.7f}'.format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            epoch_time_s = time.time() - epoch_time
            print('Epoch: {} cost time: {}'.format(epoch + 1, epoch_time_s))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion_y=criterion_y, mode_flag='vail')
            print('Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}'.format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth')))

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, relation = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_y = self._forward_batch(batch_x, relation)
                else:
                    output_y = self._forward_batch(batch_x, relation)
                batch_y_y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
                preds.append(output_y.detach().cpu().numpy())
                trues.append(batch_y_y.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        metrics_payload = {
            'schema': 'DReSTG_metrics_v1',
            'model': 'DReSTG',
            'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse),
            'mape': float(mape), 'mspe': float(mspe), 'rse': float(rse),
            'cell_identity': self._cell_identity(),
            'recipe': self._recipe(),
            'semantic_fingerprint': self._semantic_fingerprint(),
            'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == 'cuda' else None,
        }

        with open(os.path.join(folder_path, 'metrics.json'), 'w', encoding='utf-8') as fh:
            json.dump(metrics_payload, fh, indent=2, sort_keys=True)
        return metrics_payload
