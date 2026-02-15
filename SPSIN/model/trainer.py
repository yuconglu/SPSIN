import torch
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, climatology_unnormalized, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        
        self.climatology_unnormalized = torch.from_numpy(climatology_unnormalized).float().to(self.args.device)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, 'trained-best-model')
        os.makedirs(save_dir, exist_ok=True)
        best_model_filename = f'best_model_{self.args.dataset}.pth'
        self.best_path = os.path.join(save_dir, best_model_filename)

        self.optimize_horizon = getattr(args, 'optimize_horizon', 'average')

        self.best_avg_val_weighted_rmse = float('inf')
        self.best_avg_val_weighted_rmse_epoch = -1
        self.best_optimization_metric = float('inf')
        self.best_optimization_metric_epoch = -1
        self.best_avg_val_weighted_acc = float('-inf')
        self.best_acc_epoch = -1

        base_data_path = None
        if hasattr(args, 'data_dir') and args.data_dir and os.path.isdir(args.data_dir):
            base_data_path = args.data_dir
        elif hasattr(args, 'processed_data_dir') and args.processed_data_dir and os.path.isdir(args.processed_data_dir):
            base_data_path = args.processed_data_dir
        elif os.path.isfile(args.dataset):
            base_data_path = os.path.dirname(args.dataset)
        elif os.path.isdir(args.dataset):
            base_data_path = args.dataset

        if base_data_path is None or not os.path.isdir(base_data_path):
            base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

        lat_coords_filename = f'lat_coords_{args.dataset}.npy'
        lat_coords_path = os.path.join(base_data_path, lat_coords_filename)
        try:
            latitudes_np = np.load(lat_coords_path)
            self.latitudes = torch.from_numpy(latitudes_np).float().to(self.args.device)
            if self.latitudes.shape[0] != self.args.num_nodes:
                raise ValueError("Latitude coordinate mismatch")
        except FileNotFoundError:
            self.logger.error(f'Latitude coordinates file {lat_coords_path} not found')
            self.latitudes = None
        except Exception as e:
            self.logger.error(f'Error loading latitude coordinates: {e}')
            self.latitudes = None

    def get_latitude_weights(self, latitudes_tensor_1d):
        if latitudes_tensor_1d is None:
            return None
        weights = torch.cos(torch.deg2rad(latitudes_tensor_1d))
        weights = weights / weights.mean()
        return weights

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.)
                y_true_list.append(label)
                y_pred_list.append(output)
        
        y_true_scaled = torch.cat(y_true_list, dim=0)
        y_pred_scaled = torch.cat(y_pred_list, dim=0)

        y_true = self.scaler.inverse_transform(y_true_scaled)[:,:,:,:self.args.output_dim]
        if self.args.real_value:
            y_pred = y_pred_scaled[:,:,:,:self.args.output_dim]
        else:
            y_pred = self.scaler.inverse_transform(y_pred_scaled)[:,:,:,:self.args.output_dim]
        
        val_loss = self.loss(y_pred.cuda(), y_true.cuda())

        self.logger.info('Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss.item()))

        y_pred_metrics = y_pred.cpu()
        y_true_metrics = y_true.cpu()
        
        avg_val_rmse_w = float('inf')

        if self.latitudes is not None:
            latitude_weights = self.get_latitude_weights(self.latitudes.cpu())
            clim_for_acc = self.climatology_unnormalized.cpu()[:, :1].squeeze(-1)
            if latitude_weights is None:
                return val_loss.item(), avg_val_rmse_w

            val_rmse_weighted_list = []
            val_acc_weighted_list = []
            horizon_rmse_dict = {}

            num_horizon_steps = y_pred_metrics.shape[1]
            for h_idx in range(num_horizon_steps):
                y_pred_h = y_pred_metrics[:, h_idx, :, 0]
                y_true_h = y_true_metrics[:, h_idx, :, 0]

                error_h = y_pred_h - y_true_h
                weighted_squared_error_h = (error_h**2) * latitude_weights.unsqueeze(0)
                current_rmse_w = torch.sqrt(weighted_squared_error_h.mean())
                val_rmse_weighted_list.append(current_rmse_w.item())
                horizon_rmse_dict[f'h{h_idx+1}'] = current_rmse_w.item()
                
                self.logger.info('  Val Horizon {:02d}: Weighted RMSE: {:.6f}'.format(h_idx + 1, current_rmse_w.item()))

                pred_anom_h = y_pred_h - clim_for_acc.unsqueeze(0)
                true_anom_h = y_true_h - clim_for_acc.unsqueeze(0)
                
                pred_anom_prime_h = pred_anom_h - pred_anom_h.mean()
                true_anom_prime_h = true_anom_h - true_anom_h.mean()
                
                numerator = (pred_anom_prime_h * true_anom_prime_h * latitude_weights.unsqueeze(0)).sum()
                denominator_pred_sq = ((pred_anom_prime_h**2) * latitude_weights.unsqueeze(0)).sum()
                denominator_true_sq = ((true_anom_prime_h**2) * latitude_weights.unsqueeze(0)).sum()
                
                current_acc_w = numerator / (torch.sqrt(denominator_pred_sq * denominator_true_sq) + 1e-6)
                val_acc_weighted_list.append(current_acc_w.item())
                self.logger.info('  Val Horizon {:02d}: Weighted ACC:  {:.6f}'.format(h_idx + 1, current_acc_w.item()))
            
            if val_rmse_weighted_list:
                avg_val_rmse_w = np.mean(val_rmse_weighted_list)
                self.logger.info('  Validation Avg Weighted RMSE: {:.6f}'.format(avg_val_rmse_w))
                if avg_val_rmse_w < self.best_avg_val_weighted_rmse:
                    self.best_avg_val_weighted_rmse = avg_val_rmse_w
                    self.best_avg_val_weighted_rmse_epoch = epoch
            if val_acc_weighted_list:
                avg_val_acc_w = np.mean(val_acc_weighted_list)
                self.logger.info('  Validation Avg Weighted ACC:  {:.6f}'.format(avg_val_acc_w))
                if avg_val_acc_w > self.best_avg_val_weighted_acc:
                    self.best_avg_val_weighted_acc = avg_val_acc_w
                    self.best_acc_epoch = epoch

        if self.optimize_horizon == 'average':
            optimization_metric = avg_val_rmse_w
        elif self.optimize_horizon in horizon_rmse_dict:
            optimization_metric = horizon_rmse_dict[self.optimize_horizon]
        else:
            optimization_metric = avg_val_rmse_w

        return val_loss.item(), optimization_metric

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim].to(self.args.device)
            label = target[..., :self.args.output_dim].to(self.args.device)
            self.optimizer.zero_grad()

            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            
            output1, output2 = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio, apply_r_drop=True)
            
            label = self.scaler.inverse_transform(label)[:,:,:,:1]

            if not self.args.real_value:
                output1_rescaled = self.scaler.inverse_transform(output1)[:,:,:,:1]
                output2_rescaled = self.scaler.inverse_transform(output2)[:,:,:,:1]
                loss_sup1 = self.loss(output1_rescaled.cuda(), label)
                loss_sup2 = self.loss(output2_rescaled.cuda(), label)
            else:
                loss_sup1 = self.loss(output1.cuda(), label)
                loss_sup2 = self.loss(output2.cuda(), label)
            
            loss_supervised = (loss_sup1 + loss_sup2) / 2
            
            loss_reg = F.mse_loss(output1, output2)
            
            beta = self.args.r_drop_beta
            loss = loss_supervised + beta * loss_reg

            loss.backward()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        self.best_avg_val_weighted_rmse = float('inf')
        self.best_avg_val_weighted_rmse_epoch = -1
        self.best_optimization_metric = float('inf')
        self.best_optimization_metric_epoch = -1
        self.best_avg_val_weighted_acc = float('-inf')
        self.best_acc_epoch = -1
        
        not_improved_count = 0
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            
            current_avg_val_rmse_w = float('inf')

            if self.val_loader != None:
                current_val_loss, current_avg_val_rmse_w = self.val_epoch(epoch, self.val_loader)

            if self.val_loader is not None and current_avg_val_rmse_w < self.best_optimization_metric:
                self.best_optimization_metric = current_avg_val_rmse_w
                self.best_optimization_metric_epoch = epoch
                not_improved_count = 0
                
                current_best_state_dict = copy.deepcopy(self.model.state_dict())

                if not self.args.debug:
                    torch.save(current_best_state_dict, self.best_path)
                    self.logger.info(f"Saved best model to: {self.best_path}")
            else:
                not_improved_count += 1
            
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info(f"Early stopping triggered after {self.args.early_stop_patience} epochs")
                    break

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format((training_time / 60)))

    def get_best_avg_val_weighted_rmse(self):
        return self.best_optimization_metric

    def get_best_avg_val_weighted_rmse_epoch(self):
        return self.best_optimization_metric_epoch

    def test(self, model, args, data_loader, scaler, logger, path_to_model_state=None):
        if path_to_model_state != None:
            logger.info(f"Loading model state from: {path_to_model_state}")
            try:
                check_point = torch.load(path_to_model_state, map_location=args.device)
                if isinstance(check_point, dict) and 'state_dict' in check_point:
                    state_dict = check_point['state_dict']
                else:
                    state_dict = check_point
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                logger.error(f"Error loading model from {path_to_model_state}: {e}")
                return
        
        model.eval()
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim].to(args.device)
                target = target.to(args.device)
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true_list.append(label.cpu())
                y_pred_list.append(output.cpu())
        
        y_true_scaled = torch.cat(y_true_list, dim=0)
        y_pred_scaled = torch.cat(y_pred_list, dim=0)

        y_true_scaled = y_true_scaled.to(args.device)
        y_pred_scaled = y_pred_scaled.to(args.device)

        y_true = scaler.inverse_transform(y_true_scaled)[:,:,:,:args.output_dim]
        if args.real_value:
            y_pred = y_pred_scaled[:,:,:,:args.output_dim]
        else:
            y_pred = scaler.inverse_transform(y_pred_scaled)[:,:,:,:args.output_dim]

        np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())

        logger.info("Calculating metrics for test set...")
        for t_idx in range(y_true.shape[1]):
            pred_t = y_pred[:, t_idx, :, :].squeeze(-1) if args.output_dim == 1 else y_pred[:, t_idx, :, :]
            true_t = y_true[:, t_idx, :, :].squeeze(-1) if args.output_dim == 1 else y_true[:, t_idx, :, :]

            mse, mae, rmse, mape, _, _ = All_Metrics(pred_t, true_t,
                                                args.mae_thresh, args.mape_thresh)
            logger.info("  Test Horizon {:02d}: MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                t_idx + 1, mse, mae, rmse, mape*100 if mape is not None else -1))

        if self.latitudes is not None:
            latitude_weights = self.get_latitude_weights(self.latitudes.cpu())
            clim_for_acc = self.climatology_unnormalized.cpu()[:, :1].squeeze(-1)
            if latitude_weights is None:
                 logger.warning("Cannot compute weighted metrics")
                 return

            test_rmse_weighted_list = []
            test_acc_weighted_list = []
            num_horizon_steps_test = y_pred.shape[1]

            for h_idx in range(num_horizon_steps_test):
                y_pred_h_test = y_pred[:, h_idx, :, 0].cpu()
                y_true_h_test = y_true[:, h_idx, :, 0].cpu()

                error_h_test = y_pred_h_test - y_true_h_test
                weighted_squared_error_h_test = (error_h_test**2) * latitude_weights.unsqueeze(0)
                current_rmse_w_test = torch.sqrt(weighted_squared_error_h_test.mean())
                test_rmse_weighted_list.append(current_rmse_w_test.item())
                logger.info('  Test Horizon {:02d}: Weighted RMSE: {:.6f}'.format(h_idx + 1, current_rmse_w_test.item()))

                pred_anom_h_test = y_pred_h_test - clim_for_acc.unsqueeze(0)
                true_anom_h_test = y_true_h_test - clim_for_acc.unsqueeze(0)
                pred_anom_prime_h_test = pred_anom_h_test - pred_anom_h_test.mean()
                true_anom_prime_h_test = true_anom_h_test - true_anom_h_test.mean()
                
                numerator_test = (pred_anom_prime_h_test * true_anom_prime_h_test * latitude_weights.unsqueeze(0)).sum()
                denominator_pred_sq_test = ((pred_anom_prime_h_test**2) * latitude_weights.unsqueeze(0)).sum()
                denominator_true_sq_test = ((true_anom_prime_h_test**2) * latitude_weights.unsqueeze(0)).sum()
                current_acc_w_test = numerator_test / (torch.sqrt(denominator_pred_sq_test * denominator_true_sq_test) + 1e-6)
                test_acc_weighted_list.append(current_acc_w_test.item())
                logger.info('  Test Horizon {:02d}: Weighted ACC:  {:.6f}'.format(h_idx + 1, current_acc_w_test.item()))
            
            if test_rmse_weighted_list:
                avg_test_rmse_w = np.mean(test_rmse_weighted_list)
                logger.info('  Test Avg: Weighted RMSE: {:.6f}'.format(avg_test_rmse_w))
            if test_acc_weighted_list:
                avg_test_acc_w = np.mean(test_acc_weighted_list)
                logger.info('  Test Avg: Weighted ACC:  {:.6f}'.format(avg_test_acc_w))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        return k / (k + math.exp(global_step / k))
