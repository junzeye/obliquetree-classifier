# scikit-obliquetree-classifier

<div align="center">
Oblique Decision Tree in Python
</div>

## Introduction

Decision trees are a popular machine learning method that is highly interpretable. The vast majority of publicly available decision tree libraries implements an axis-parallel version of decision trees, where splits at each decsion node involve only a single feature variable. This leaves room for performance improvement because we might learn patterns in a dataset better by considering oblique decision boundaries.

In comparison to axis-parallel trees, oblique decision trees partition a feature space by drawing half-spaces involving all feature variables. However, despite much research showing the exceptional performance of oblique decision trees, there is the lack of an open-source package that implements an oblique decision tree classificaton algorithm.

This gap in technical infrastructure motivates us to program and publish the Python implementation of the HHCART algorithm (Wickramarachchi et al. 2016) for __classification tasks__. We hope that this repository will be a handy tool for researchers data scientists who want to leverage the increased representation power of oblique decision trees.

## 🚀 Features
* A simple scikit-learn interface for oblique decision tree classifiers
* Provides a wrapper class in `HHCART_vis.py` to allow for convenient tree visualization

## Example
You can find a more detailed example in the Jupyter notebook `example.ipynb`.
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from HHCART import HouseHolderCART  # oblique tree classifier
from segmentor import Gini, TotalSegmentor  # module to determine splits
import numpy as np
import itertools, time

# Load training data - we use the Iris dataset as an example
X, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

# Initialize an HHCART classifier object
sgmtr = TotalSegmentor()
HHTree = HouseHolderCART(impurity = Gini(), segmentor = sgmtr, max_depth = 5, 
                                    min_samples = 4)
# max_depth: maximum depth of the decision tree. 
# min_samples: minimum allowed number of samples in a terminal node

# Train the classifier
HHTree.fit(x_train, y_train)

# Evaluate the classifier performance
train_score = accuracy_score(y_train, HHTree.predict(x_train))
test_score = accuracy_score(y_test, HHTree.predict(x_test))
print(f"train accuracy: {train_score:.00%}")
print(f"test accuracy: {test_score:.00%}")
```

## 🛡 License

[![License](https://img.shields.io/github/license/zhenlingcn/scikit-obliquetree)](https://github.com/zhenlingcn/scikit-obliquetree/blob/master/LICENSE)

This project is licensed under the terms of the `Apache Software License 2.0` license. See [LICENSE](https://github.com/zhenlingcn/scikit-obliquetree/blob/master/LICENSE) for more details.

## 📃 Bibliography
```
@article{WICK201612,
title = {HHCART: An oblique decision tree},
journal = {Computational Statistics & Data Analysis},
volume = {96},
pages = {12-23},
year = {2016},
issn = {0167-9473},
url = {https://www.sciencedirect.com/science/article/pii/S0167947315002856},
author = {D.C. Wickramarachchi and B.L. Robertson and M. Reale and C.J. Price and J. Brown},
keywords = {Oblique decision tree, Data classification, Statistical learning, Householder reflection, Machine learning}
}

@misc{scikit-obliquetree,
  author = {ECNU},
  title = {Oblique Decision Tree in Python},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zhenlingcn/scikit-obliquetree}}
}
```

## Acknowledgements

I would like to thank Github user `hengzhe-zhang` (2nd citation) for generously sharing the algorithm they developed for growing decision trees to complete regression tasks. My work generalizes his code by enriching his code to generate trees that can also complete classification tasks. You can refer to `hengzhe-zhang`'s repo from [here](https://github.com/zhenlingcn/scikit-obliquetree).

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template).
