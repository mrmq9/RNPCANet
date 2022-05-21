The official python implementation of RNPCANets in the paper [Randomized Non-linear PCA Networks](https://www.sciencedirect.com/science/article/pii/S0020025520307635)

## Requirements

* python==3.9.7
* ruamel.yaml==0.17.21
* numpy==1.20.3
* scipy==1.7.1
* scikit-learn==0.24.2
* pytorch==1.9.0
* torchvision==0.10.0
* tqdm==4.62.2

## Datasets

The configurations for three datasets, MNIST, coil-20, and coil-100 are available in the code.
The MNIST dataset will be downloaded automatically once the code is run by --dataset mnist.

To download coil-20 or coil-100 datasets, run the following command in the data folder:
```bash
bash download.sh <dataset>
```

## Train and evaluation

Run the following command for training and evaluation:
```bash
python main.py --dataset <dataset>
```
The results on coil-100 should be better than what is reported in the paper due to a different patch mean removal.

## Reference
[1] Qaraei et al., [Randomized Non-linear PCA Networks](https://www.sciencedirect.com/science/article/pii/S0020025520307635), Information Sciences, 2021.
