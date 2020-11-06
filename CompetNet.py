import numpy as np


class CompetNet:

    def __init__(self, num_inputs, num_compets, init_weights=None, norm=True):
        """
        :param num_inputs: number of input neurons
        :param num_compets: number of compet neurons
        :param init_weights: init val for compet neurons. shape: (`num_inputs`, `num_compets`)
        :param norm: norm inputs and weights or not
        """
        self.num_inputs = num_inputs
        self.num_compets = num_compets
        self.norm = norm
        if init_weights is not None:
            if np.shape(init_weights) != (num_inputs, num_compets):
                raise ValueError('Invalid shape for `init_weights`.')
            self._w = np.float32(init_weights)
        else:
            self._w = np.random.uniform(-1, 1, size=(num_inputs, num_compets)).astype(np.float32)

    def __call__(self, xdata):
        """
        :param xdata: data for training. shape: (`num_inputs`, `batch_size`)
        :return:
        """
        return self.predict(xdata)

    def train(self, xdata, lr=0.1):
        """
        :param xdata: data for training. shape: (`num_inputs`, `batch_size`)
        :param lr: learning rate
        :return:
        """
        outs = self.predict(xdata)
        dims, nums = np.shape(xdata)
        for n in range(nums):
            self._w[:, outs[n]] += lr * (xdata[:, n] - self._w[:, outs[n]])

        return outs

    def predict(self, xdata):
        """
        :param xdata: data for training. shape: (`num_inputs`, `batch_size`)
        :return:
        """
        dims, nums = np.shape(xdata)
        if dims != self.num_inputs:
            raise ValueError('Invalid shape for training data.')
        if self.norm:
            xdata = self.__norm_vector(xdata)
            self._w = self.__norm_vector(self._w)
            compet = np.matmul(np.transpose(self._w), xdata)
            outs = np.argmax(compet, axis=0)
        else:
            xdata = np.expand_dims(xdata, axis=1)
            xdata = np.tile(xdata, [1, self.num_compets, 1])
            wd = np.expand_dims(self._w, axis=2)
            wd = np.tile(wd, [1, 1, xdata.shape[-1]])
            compet = xdata - wd
            compet = np.square(compet)
            compet = np.sum(compet, axis=0)
            outs = np.argmin(compet, axis=0)

        return outs

    def get_weights(self):
        return self._w

    @staticmethod
    def __norm_vector(vector):
        square = np.square(vector)
        square = np.sum(square, axis=0, keepdims=True)
        return vector / np.sqrt(square)



