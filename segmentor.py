from abc import ABCMeta, abstractmethod
from collections import Counter
import numpy as np

class Gini:
    """
    Gini impurity criterion
    """ 
    def __call__(self, left_label, right_label):
        '''
        `all_labels` represents all possible class labels in the dataset
        '''
        key_freq_left = np.array(list(Counter(left_label).values()))
        key_freq_right = np.array(list(Counter(right_label).values()))
        left_len, right_len = sum(key_freq_left), sum(key_freq_right)
        total = left_len + right_len

        Gini_left = 1.0 - sum ((key_freq_left / left_len) ** 2)
        Gini_right = 1.0 - sum ((key_freq_right / right_len) ** 2)

        return (left_len / float(total)) * Gini_left + (right_len / float(total)) * Gini_right
        
class MSE:
    """
    Mean squared error impurity criterion
    """

    def __call__(self, left_label, right_label):
        left_len, right_len = len(left_label), len(right_label)

        left_std = np.std(left_label)
        right_std = np.std(right_label)

        total = left_len + right_len

        return (left_len / total) * left_std + (right_len / total) * right_std


class SegmentorBase:
    """
    Abstract segmentor class. Segmentor called in nodes for find best split.


    Parameters
    -----------

    msl : int, optional (default=1)
    The minimum number of samples required to be at a leaf nodes.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def _split_generator(self, X):
        """
        Abstract method for split.


        Parameter
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.
        """
        pass

    def __init__(self, msl=1):
        self._min_samples_leaf = msl

    def __call__(self, X, y, impurity=Gini()):
        """
        Parameters
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.
        y : array-like, shape = [n_samples]
        The target values.
        impurity : object of impurity (default=Gini()). The name of criterion 
        to measure the quality of a split. Use MSE() for regression tree

        Returns
        -----------

        Tuple of the following elements:
        best_impurity : float.
        The best value of impurity.
        best_split_rule : tuple.
        The pair of feature and value.
        best_left_i : numpy.ndarray.
        The indexes of left node objects.
        best_right_i : numpy.ndarray.
        The indexes of right node objects.
        """
        best_impurity = float("inf")
        best_split_rule = None
        best_left_i = None
        best_right_i = None
        splits = self._split_generator(X)

        for left_i, right_i, split_rule in splits:
            if (
                left_i.size >= self._min_samples_leaf
                and right_i.size >= self._min_samples_leaf
            ):
                left_labels, right_labels = y[left_i], y[right_i]
                cur_impurity = impurity(left_labels, right_labels)
                if cur_impurity < best_impurity:
                    best_impurity = cur_impurity
                    best_split_rule = split_rule
                    best_left_i = left_i
                    best_right_i = right_i
        return (best_impurity, best_split_rule, best_left_i, best_right_i)


class MeanSegmentor(SegmentorBase):
    """
    Split based on mean value of each feature.
    """

    def _split_generator(self, X):
        """
        Parameters
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.


        Returns
        -----------

        Tuple of the following elements:
        left_i : numpy.ndarray.
        The indexes of left node objects.
        right_i : numpy.ndarray.
        The indexes of right node objects.
        split_rule : tuple.
        The pair of feature and value.
        """
        for feature_i in range(X.shape[1]):
            feature_values = X[:, feature_i]
            mean = np.mean(feature_values)
            left_i = np.nonzero(feature_values < mean)[0]
            right_i = np.nonzero(feature_values >= mean)[0]
            split_rule = (feature_i, mean)
            yield (left_i, right_i, split_rule)


class TotalSegmentor(SegmentorBase):
    """
    Generates all possible axis parallel splits.
    """

    def _split_generator(self, X):

        for feature_i in range(X.shape[1]):
            feature_values = X[:, feature_i]
            for val in feature_values:
                left_i = np.nonzero(feature_values < val)[0]
                right_i = np.nonzero(feature_values >= val)[0]
                split_rule = (feature_i, val)
                yield (left_i, right_i, split_rule)