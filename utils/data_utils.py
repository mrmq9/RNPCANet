# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import os
from PIL import Image
import numpy as np
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

def read_mnist(data_path, train=True):
    data_labels = datasets.MNIST(root=data_path, train=train,
                                 download=True, transform=None)
    data = np.asarray(data_labels.data/255)
    labels = np.asarray(data_labels.targets)
    return data, labels

def read_coil(data_path, images_name=None):
    if images_name is None:
        images_name = os.listdir(data_path)
    data, labels = [], []
    for img_name in images_name:
        if img_name.startswith('obj'):
            labels.append(int(img_name[3:].split('__')[0]))
            img = Image.open(os.path.join(data_path, img_name))
            img = np.asarray(img)/255
            if img.ndim==3:
                img = img.transpose(2, 0, 1)
            data.append(img)
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels


def data_split(data, labels):
    '''
        random splitting coil dataset
    '''
    unique_labels = np.unique(labels)
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    for l in unique_labels:
        ind_label = np.where(labels == l)[0]
        num_samples = len(ind_label)
        tr_data, te_data, tr_labels, te_labels = train_test_split(
            data[ind_label, :, :], labels[ind_label], train_size=round(num_samples/6))
        train_data.append(tr_data)
        test_data.append(te_data)
        train_labels.append(tr_labels)
        test_labels.append(te_labels)

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    train_labels = np.concatenate(train_labels)
    test_labels = np.concatenate(test_labels)

    return train_data, test_data, train_labels, test_labels


def get_data_labels(args):
    
    if args.dataset == 'coil-20' or args.dataset == 'coil-100':
        
        with open(os.path.join(args.data_dir,
                  F'data-split/tr-{args.dataset}.txt'), 'r') as f:
            tr_names = f.read().split('\n')

        with open(os.path.join(args.data_dir,
                  F'data-split/te-{args.dataset}.txt'), 'r') as f:
            te_names = f.read().split('\n')

        data_path = os.path.join(args.data_dir, args.dataset)
        train_data, train_labels = read_coil(data_path, tr_names)
        test_data, test_labels = read_coil(data_path, te_names)
    
    if args.dataset == 'mnist':
        train_data, train_labels = read_mnist(args.data_dir, train=True)
        test_data, test_labels = read_mnist(args.data_dir, train=False)

    # train_data, test_data, train_labels, test_labels = data_split(data, labels)

    return train_data, test_data, train_labels, test_labels
