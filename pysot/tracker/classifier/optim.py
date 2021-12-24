import torch
from pysot.tracker.classifier.libs import optimization, TensorList, operation
import math
from pysot.core.config import cfg


class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList,
                 filter_reg: torch.Tensor, projection_reg, sample_weights: TensorList,
                 projection_activation, att_activation, response_activation):

        self.training_samples = training_samples
        self.y = y
        self.sample_weights = sample_weights
        self.filter_reg = filter_reg
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.att_activation = att_activation
        self.response_activation = response_activation

        self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):

        filter = x[:len(x)//2]  # w2 in paper
        P = x[len(x)//2:]       # w1 in paper

        # Compression module
        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)
        # Filter module
        residuals = operation.conv2d(compressed_samples, filter, mode='same').apply(self.response_activation)
        residuals = residuals - self.y
        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)

        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals


    def ip_input(self, a: TensorList, b: TensorList):

        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]

        # Filter inner product
        # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)

        # Add projection matrix part
        # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
        ip_out += operation.conv2d(a_P.view(1, -1, 1, 1), b_P.view(1, -1, 1, 1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())

    def M1(self, x: TensorList):
        # factorized convolution
        return x / self.diag_M

class ConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y: TensorList, filter_reg: torch.Tensor, sample_weights: TensorList, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        # Do convolution and compute residuals
        residuals = operation.conv2d(self.training_samples, x, mode='same').apply(self.response_activation)
        residuals = residuals - self.y
        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)
