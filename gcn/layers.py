import tensorflow as tf


class GCNLayer(object):
    """
    Basic Graph Convolutional Network Layer
    """

    def __init__(self, feats, units, adj=None, deg=None, support=1,
                 kernel_initializer=tf.variance_scaling_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 activation=tf.nn.relu):
        """
        :param feats: Input features for graph with a shape of (?, ?, 1)
        :param units: Number of units for the output of the kernel
        :param adj: Optional Adjacency matrix if not applied before
        :param deg: Optional Degree matrix if not applied before
        :param support: number of support values that get summed in the activation function
        :param kernel_initializer: Kernel weight initializer
        :param bias_initializer: Bias Initializer
        :param activation: activation
        """
        if support < 1:
            raise RuntimeError("Filters cannot be less than 1")
        self.weights = []
        self.bias = tf.Variable(bias_initializer((1, units)))
        for i in range(support):
            self.weights.append(tf.Variable(kernel_initializer((int(feats.shape[-1]), units))))
        self.adj = adj
        self.deg = deg
        self.feats = feats
        self.act = activation

    def __call__(self, *args, **kwargs):
        """
        Apply GCN Layer
        """
        total = []
        for i in range(len(self.weights)):
            weights = self.weights[i]
            A_D = None
            if self.adj and self.deg:
                A_D = tf.matmul(self.deg, tf.matmul(self.adj, self.deg))
                A_D = tf.tile(tf.expand_dims(A_D, 0), (tf.shape(self.feats)[0], 1, 1))
            wts = tf.tile(tf.expand_dims(weights, 0), (tf.shape(self.feats)[0], 1, 1))
            kernel = tf.matmul(self.feats, wts)
            if A_D:
                total.append(tf.matmul(A_D, kernel))
            else:
                total.append(kernel)

        total_add = tf.add_n(total)
        return self.act(total_add + self.bias)
