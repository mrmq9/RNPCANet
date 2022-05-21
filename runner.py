# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import numpy as np
import torch
from torch.nn import parameter
from torch.distributions import MultivariateNormal, Uniform
from model import RNPCANet
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RnRunner(object):
    def __init__(self, **kwargs):
        self.model = RNPCANet(**kwargs)
        self.clf = None
        self.sigma1 = kwargs['sigma1']
        self.sigma2 = kwargs['sigma2']
        self.kernel_size1 = kwargs['kernel_size1']
        self.kernel_size2 = kwargs['kernel_size2']
        self.rff_num_filter1 = kwargs['rff_num_filter1']
        self.rff_num_filter2 = kwargs['rff_num_filter2']
        self.pca_num_filter1 = kwargs['pca_num_filter1']
        self.pca_num_filter2 = kwargs['pca_num_filter2']

    def train_network(self, data_loader):
        with torch.no_grad():

            # set the RFF weights
            self.model.rff1.weight, self.model.rff1.bias = self.get_rff_weights(
                self.sigma1, self.kernel_size1, self.rff_num_filter1,
                self.model.rff1.weight.shape)

            self.model.rff2.weight, self.model.rff2.bias = self.get_rff_weights(
                self.sigma2, self.kernel_size2, self.rff_num_filter2,
                self.model.rff2.weight.shape)

            self.model.to(DEVICE)

            for level in range(2):
                if level==0:
                    Rx = torch.zeros((self.rff_num_filter1, self.rff_num_filter1)).to(DEVICE)
                else:
                    Rx = torch.zeros((self.rff_num_filter2, self.rff_num_filter2)).to(DEVICE)
                num_patch = 0
                for step, batch in enumerate(tqdm(data_loader, desc=F'Training level {level+1} of RNPCANet')):
                    data = batch[0].to(DEVICE)
                    if data.dim() == 3: # in case of gray scale data, add one dimension to data
                        data = data.unsqueeze(1)
                    if level == 0:
                        data = self.model(data, return_loc='rff1')
                    else:
                        data = self.model(data, return_loc='rff2')
                    data = data.permute(1,0,2,3).reshape(data.shape[1], data.shape[0]*data.shape[2]*data.shape[3])
                    num_patch += data.shape[1]
                    Rx += torch.mm(data, data.T)
                Rx = Rx / num_patch
                eig_val, eig_vec = torch.linalg.eigh(Rx)
                _, large_eig_idx = torch.sort(eig_val, descending=True)
                if level == 0:
                    pca_filters = eig_vec[:, large_eig_idx[0:self.pca_num_filter1]]
                    self.model.pca1.weight = parameter.Parameter(pca_filters.T).requires_grad_(False)
                else:
                    pca_filters = eig_vec[:, large_eig_idx[0:self.pca_num_filter2]]
                    self.model.pca2.weight = parameter.Parameter(pca_filters.T).requires_grad_(False)
    

    def get_rff_weights(self, sigma, kernel_size, rff_num_filter, weights_shape):
        '''
            return the weights of RFF layers by sampling from 
            a normal distribution and uniform sampling for bias
        '''
        normal_dist = MultivariateNormal(torch.zeros(weights_shape[1]*kernel_size**2),
                                         covariance_matrix=(1/sigma**2)*torch.eye(weights_shape[1]*kernel_size**2))
        unifrom = Uniform(0.0, 2.0*torch.pi)

        weights = parameter.Parameter(normal_dist.rsample(
            [rff_num_filter]).reshape(weights_shape)).requires_grad_(False)
        bias = parameter.Parameter(unifrom.sample(
            [rff_num_filter])).requires_grad_(False)

        return weights, bias


    def train_svm(self, data_loader):
        with torch.no_grad():
            data_all, labels_all = [], [] 
            for step, batch in enumerate(tqdm(data_loader, desc='Extracting features of training data')):
                data = batch[0].to(DEVICE)
                labels = batch[1]

                if data.dim() == 3:
                        data = data.unsqueeze(1)

                data = self.model(data)
                data_all.append(data)
                labels_all.extend(list(map(int, labels)))
            
            data_all = vstack(data_all)
            labels_all = np.array(labels_all)
        
        print("******** Training SVM ********")
        self.clf = LinearSVC(fit_intercept=False, max_iter=300)
        self.clf.fit(data_all, labels_all)
    

    def predict_svm(self, data_loader, eval_step):
        
        print('**** Evaluating the model ****')
        with torch.no_grad():
            data_all, labels_all = [], []
            acc, num_data = 0.0, 0
            for step, batch in enumerate(data_loader, 1):
                data = batch[0].to(DEVICE)
                labels = batch[1]

                if data.dim() == 3: # in case of gray scale data, add one dimension to data
                    data = data.unsqueeze(1)

                data = self.model(data)
                data_all.append(data)
                labels_all.extend(list(map(int, labels)))

                if step%eval_step==0 or step==len(data_loader):
                    print(F'Evaluation results after {step} steps')
                    data_all = vstack(data_all)
                    labels_all = np.array(labels_all)
                    num_data += labels_all.shape[0]

                    preds = self.clf.predict(data_all)
                    acc += np.sum(preds==labels_all)
                    print(F'Accuracy: {round(acc*100.0/num_data, 2)}')

                    data_all, labels_all = [], []

            print(F'Total accruacy: {round(acc*100.0/num_data,2)}')
