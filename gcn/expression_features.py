"""
Author: Derek Miller

Sources:
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
"""

import numpy as np


class CorrelationExpressionGraph(object):
    """
    Converts pandas dataframe to features ready for training for gene expression data or any other graph using
    correlation edges.

    This class wants a dataframe like:

    |label|gene_1|gene_2|gene_3
    ---------------------------
     high, 100.0, 200.0, 300.0
    """

    def __init__(self, df, label_column, is_regression=False, feature_columns=None, ignore_columns=None):
        """
        :param df: Pandas dataframe
        :param label_column: Label Column in the pandas dataframe
        :param is_regression: Specifies if it is a regression problem
        :param feature_columns: Specific feature columns (if not all)
        :param ignore_columns: Columns to ignore in the dataframe
        """
        self.df = df
        self.label_column = label_column

        if feature_columns:
            self.other_columns = feature_columns
        else:
            to_ignore = [label_column]
            if ignore_columns:
                to_ignore += ignore_columns
            self.other_columns = list(set(to_ignore).symmetric_difference(set(df.columns.values)))
        self.is_regression = is_regression

    def __build_labels(self, replace_values):
        labels = self.df[self.label_column].values
        if replace_values:
            for k, v in replace_values.items():
                labels[labels == k] = v
        if not self.is_regression:
            self.num_labels = int(labels.max())
            nu_labels = np.zeros((len(labels), self.num_labels + 1))
            for i in range(0, len(labels)):
                nu_labels[i][int(labels[i])] = 1.0
            return nu_labels.astype(np.float32)
        return labels.astype(np.float32)

    def __handle_adjacency(self, features, correlation_bin):
        """
        Expression features adjacency matrix. Note, the identity matrix is not needed as each correlation value is 100%
        correlated with itself
        :param features: Features from the expression values
        :param correlation_bin: Makes the correlation edges binary values instead of continuous.
        :return Adjacency matrix
        """

        transposed_features = features.transpose()
        adjacency_matrix = np.corrcoef(transposed_features)
        if correlation_bin:
            adjacency_matrix[adjacency_matrix >= correlation_bin] = 1.0
            adjacency_matrix[adjacency_matrix <= correlation_bin] = 0.0
        return adjacency_matrix

    def __handle_diagonal(self, adjacency_matrix, apply_diagonal_normalization=True):
        """
        Applies the normalization on the Diagonal
        :param adjacency_matrix: Adjacency matrix from expression values
        :param apply_diagonal_normalization: Use the normalization from the original Kiph paper
        :return: Diagonal
        """

        D = np.array(np.sum(adjacency_matrix, axis=0))
        if apply_diagonal_normalization:
            D = np.power(np.array(np.diag(D)), -0.5)
        else:
            D = np.array(np.diag(D))
        D[np.isinf(D)] = 0.0
        D[np.isnan(D)] = 0.0
        return D

    def build_features(self, replace_labels=None, correlation_bin=None,
                       apply_diagonal_normalization=True):
        """
        :param replace_labels: If your labels are strings, you can replace them with a dictionary.
        For example: {'high': 1.0, 'low': 0.0} will replace labels with 'high' to 1.0
        :param correlation_bin: number to use for binary bins (for example if set to 0.5, you want values greater than
        0.5 to be 1.0 and numbers less than 0.5 to be 0.0.
        :param apply_diagonal_normalization: Apply diagonal normalization from the original GCN paper
        :return: features, labels, A, and D
        """

        labels = self.__build_labels(replace_labels)
        features = self.df[self.other_columns].values.astype(np.float32)
        full_features = features.reshape((features.shape[0], features.shape[-1], 1))
        adjacency_matrix = self.__handle_adjacency(features, correlation_bin)
        D = self.__handle_diagonal(adjacency_matrix, apply_diagonal_normalization)
        return full_features, labels, adjacency_matrix, D

    def build_features_apply_mult(self, replace_labels=None, correlation_bin=None,
                                  apply_diagonal_normalization=True):
        """
        Instead of passing the features to the GCN for this constant computation, it handles the multiplication here.
        :param replace_labels: If your labels are strings, you can replace them with a dictionary.
        For example: {'high': 1.0, 'low': 0.0} will replace labels with 'high' to 1.0
        :param correlation_bin: number to use for binary bins (for example if set to 0.5, you want values greater than
        0.5 to be 1.0 and numbers less than 0.5 to be 0.0.
        :param apply_diagonal_normalization: Apply diagonal normalization from the original GCN paper
        :return: features with dot product of Diagonal and Adjacency matrix and labels
        """

        feats, labels, A, D = self.build_features(replace_labels, correlation_bin, apply_diagonal_normalization)
        A_D = np.dot(D, A)
        A_D = np.tile(A_D, (feats.shape[0], 1, 1))
        # Formula D^-0.5 * A * D^-0.5 * X
        return A_D * feats, labels

    def build_features_with_adjacency(self, replace_labels=None, correlation_bin=None,
                                      apply_diagonal_normalization=True):
        feats, labels, A, D = self.build_features(replace_labels, correlation_bin, apply_diagonal_normalization)
        A_D = np.dot(D, A)
        return feats, labels, A_D