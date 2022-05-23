# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import os
import argparse
from utils import get_data_labels
from utils.dataset import RnDataset
from torch.utils.data import DataLoader
from runner import RnRunner
import ruamel.yaml as yaml


def main(args):
    train_data, test_data, train_labels, test_labels = get_data_labels(args)

    train_dataset = RnDataset(train_data, train_labels)
    test_dataset = RnDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_train)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_test)


    with open(os.path.join(args.config_dir, F'{args.dataset}.yaml'), 'r') as f:
        hyper_params = yaml.safe_load(f)

    runner = RnRunner(**hyper_params)
    runner.train_network(train_loader)
    runner.train_svm(train_loader)

    runner.predict_svm(test_loader, eval_step=args.eval_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/', type=str,
                         help='path of the datasets')
    parser.add_argument('--config_dir', default='configure/', type=str,
                         help='path of the configurations')
    parser.add_argument('--dataset', required=True, type=str,
                         choices=['mnist', 'coil-20', 'coil-100'], help='name of the dataset')
    parser.add_argument('--batch_train', default=1, type=int,
                         help='batch size for training')
    parser.add_argument('--batch_test', default=1, type=int,
                         help='batch size for evaluating')
    parser.add_argument('--eval_step', default=1000, type=int,
                         help='step for evaluation')

    args = parser.parse_args()

    main(args)
