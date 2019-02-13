# Code for the visualization experiment of "Dimensionality Reduction for Representing the Knowledge of Probabilistic Models" (ICLR 2019)

The code was tested on pytorch 0.3.0

Run the following code to train low-dimensionality representations for the test set of CIFAR-100:

```
 python DRPR_train.py
```

The dataset name can be changed with the variable called *dataset* which can take the following values: "cifar10", "cifar100", "mnist" or "stl10".

It saves the low-dimensional representations (2d by default) in the created folder *dataset_learned_representations* where *dataset* is the name of the dataset.

## Dataset folders

Each dataset folder "cifar10", "cifar100", "mnist" and "stl10" contains the following files:

- *original_data.txt* contains the output representations of a CNN (before the softmax activation).
- *proba_data_tau.npy* where *tau* is the value of the temperature. More details can be found in the appendix of the paper (Section C.1.2). It corresponds to applying the script called "proba_extraction.m" in the *scripts* folder.
- *labels.txt* contains the labels of the examples.

## Plotting the learned low-dimensional representations

We provide plot functions in matlab in the *scripts* folder (their name prefix is *plot*).
