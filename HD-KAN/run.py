import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
    # Visualization
    parser.add_argument('--enable_visual', action='store_true', default=True,
                        help='Enable visualization during testing for result analysis and periodic pattern inspection')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save model')
    parser.add_argument('--percent', type=int, default=100, help='Percentage of training data')


    parser.add_argument('--patch_len', type=int, default=12, help='patch_len')
    parser.add_argument('--degree1', type=int, default=6, help='KAN degree 1')
    parser.add_argument('--degree2', type=int, default=3, help='KAN degree 2')


    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='HDKAN',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')



    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')


    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--alpha', type=float, default=0.35, help='Weight of time-frequency domain MAE loss component')
    parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--inverse', action='store_true', help='Apply inverse transformation to output predictions',
                        default=False)

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):

            # setting record of experiments
            setting = '{}_{}_{}_ft{}_bz{}_sl{}_pl{}_lr{}_dm{}_d1{}_d2{}_patl{}_dr{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.batch_size,
                args.seq_len,
                args.pred_len,
                args.learning_rate,
                args.d_model,
                args.degree1,
                args.degree2,
                args.patch_len,
                args.dropout)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_bz{}_sl{}_pl{}_lr{}_dm{}_d1{}_d2{}_patl{}_dr{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.batch_size,
            args.seq_len,
            args.pred_len,
            args.learning_rate,
            args.d_model,
            args.degree1,
            args.degree2,
            args.patch_len,
            args.dropout)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()