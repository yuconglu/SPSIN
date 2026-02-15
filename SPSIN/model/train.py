import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.spsin_model import SPSIN

from model.trainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch


Mode = 'train'
DEBUG = False
DATASET = 'GIS'    
DEVICE = 'cuda'
MODEL = 'SPSIN'

script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, '{}.conf'.format(DATASET))
config = configparser.ConfigParser()
config.read(config_file, encoding='utf-8')


def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss


def run_training():
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('--dataset', default=DATASET, type=str)
    args.add_argument('--mode', default=Mode, type=str)
    args.add_argument('--device', default=DEVICE, type=str)
    args.add_argument('--debug', default=DEBUG, type=eval)
    args.add_argument('--model', default=MODEL, type=str)
    args.add_argument('--cuda', default=True, type=bool)

    args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('--lag', default=config['data']['lag'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--tod', default=config['data']['tod'], type=eval)
    args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('--feature_wise', default=config['data']['feature_wise'], type=eval)
    args.add_argument('--data_dir', default=config['data'].get('data_dir', None), type=str)
    args.add_argument('--height', default=config['data'].getint('height', 32), type=int)
    args.add_argument('--width', default=config['data'].getint('width', 64), type=int)

    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--hidden_dim', default=config['model']['hidden_dim'], type=int)
    args.add_argument('--alpha', default=config['model']['alpha'], type=float)
    args.add_argument('--time_dependence', default=config['model']['time_dependence'], type=eval)
    args.add_argument('--time_divided', default=config['model']['time_divided'], type=eval)
    args.add_argument('--model_type', default=config['model']['model_type'], type=str)
    args.add_argument('--encoder_type', default=config['model'].get('encoder_type', 'PIDC'), type=str)
    args.add_argument('--imex_iters', default=config['model'].getint('imex_iters', 2), type=int)
    args.add_argument('--use_second_order', default=config['model'].getboolean('use_second_order', False), type=eval)
    args.add_argument('--use_pidc_corrector', default=config['model'].getboolean('use_pidc_corrector', True), type=eval)

    args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('--seed', default=config['train']['seed'], type=int)
    args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('--epochs', default=config['train']['epochs'], type=int)
    args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('--teacher_forcing', default=False, type=bool)
    args.add_argument('--real_value', default=config['train']['real_value'], type=eval)
    args.add_argument('--warm_start', default=config['train'].getboolean('warm_start', False), type=eval)
    args.add_argument('--optimize_horizon', default=config['train'].get('optimize_horizon', 'average'), type=str)

    args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)

    args.add_argument('--log_dir', default='./', type=str)
    args.add_argument('--log_step', default=config['log']['log_step'], type=int)
    args.add_argument('--plot', default=config['log']['plot'], type=eval)

    args.add_argument('--dropout_rate', type=float, default=config['train'].getfloat('dropout_rate', 0.1))
    args.add_argument('--r_drop_beta', type=float, default=config['train'].getfloat('r_drop_beta', 1.0))
    args.add_argument('--num_afno_blocks', type=int, default=config['model'].getint('num_afno_blocks', 8))

    parsed_args = args.parse_known_args()[0]

    init_seed(parsed_args.seed)

    if torch.cuda.is_available() and parsed_args.cuda:
        parsed_args.device = 'cuda'
    else:
        parsed_args.device = 'cpu'

    if parsed_args.time_dependence:
        parsed_args.input_dim = parsed_args.input_dim + 1

    train_loader, val_loader, test_loader, scaler, climatology_unnormalized = get_dataloader(
        parsed_args,
        normalizer=parsed_args.normalizer,
        tod=parsed_args.tod, 
        dow=False,
        weather=False, 
        single=False
    )

    from lib.load_dataset import get_adjacency_matrix
    edge_index = get_adjacency_matrix(parsed_args)

    model = SPSIN(parsed_args, edge_index)
    model = model.to(parsed_args.device)

    if parsed_args.warm_start:
        model_path = os.path.join(script_dir, 'trained-best-model', f'best_model_{parsed_args.dataset}.pth')
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=parsed_args.device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained model from: {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load pretrained model ({e}). Using random initialization.")
                for p in model.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.uniform_(p)
        else:
            print(f"Warning: Pretrained model not found at {model_path}. Using random initialization.")
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    print_model_parameters(model, only_num=False)

    if parsed_args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif parsed_args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(parsed_args.device)
    elif parsed_args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(parsed_args.device)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(params=model.parameters(), lr=parsed_args.lr_init, eps=1.0e-8,
                                 weight_decay=0.0005, amsgrad=False)
    lr_scheduler = None
    if parsed_args.lr_decay:
        lr_decay_steps = [int(i) for i in list(parsed_args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=parsed_args.lr_decay_rate)

    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', parsed_args.dataset, current_time)
    os.makedirs(log_dir_path, exist_ok=True)
    parsed_args.log_dir = log_dir_path

    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                      parsed_args, climatology_unnormalized, lr_scheduler=lr_scheduler)

    if parsed_args.mode == 'train':
        trainer.train()
    elif parsed_args.mode == 'test':
        logger = trainer.logger
        model_path = os.path.join(script_dir, 'trained-best-model', f"best_model_{parsed_args.dataset}.pth")
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            return

        logger.info(f"Loading model from: {model_path}")
        trainer.test(model, parsed_args, test_loader, scaler, logger, path_to_model_state=model_path)
    else:
        raise ValueError(f"Unknown mode: {parsed_args.mode}")


if __name__ == '__main__':
    run_training()
