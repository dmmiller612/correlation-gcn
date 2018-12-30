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
            if self.adj is not None:
                if self.deg is not None:
                    A_D = tf.matmul(self.deg, tf.matmul(self.adj, self.deg))
                else:
                    A_D = self.adj
                if len(self.adj.shape) == 2:
                    A_D = tf.tile(tf.expand_dims(A_D, 0), (tf.shape(self.feats)[0], 1, 1))
            wts = tf.tile(tf.expand_dims(weights, 0), (tf.shape(self.feats)[0], 1, 1))
            kernel = tf.matmul(self.feats, wts)
            if A_D is not None:
                total.append(tf.matmul(A_D, kernel))
            else:
                total.append(kernel)

        total_add = tf.add_n(total)
        return self.act(total_add + self.bias)


class GraphDiffPool(object):

    """
    My implementation based on the paper here https://arxiv.org/pdf/1806.08804.pdf .
    The paper is vague on some mathematic details, but I think this gets the gist.
    """

    def __init__(self, feats, units, adj, pool_size=10, support=1,
                 kernel_initializer=tf.variance_scaling_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 activation=tf.nn.relu):
        if support < 1:
            raise RuntimeError("Filters cannot be less than 1")
        self.embed_weights = []
        self.bias = tf.Variable(bias_initializer((1, units)))
        for i in range(support):
            self.embed_weights.append(tf.Variable(kernel_initializer((int(feats.shape[-1]), units))))
        self.pool_weights = tf.Variable(kernel_initializer((int(feats.shape[-1]), pool_size)))
        self.pool_bias = tf.Variable(bias_initializer((1, pool_size)))
        self.adj = adj
        self.feats = feats
        self.act = activation
        self.pool_size = pool_size

    def __apply_mat_mul(self, weights):
        if len(self.adj.shape) == 2:
            A_D = tf.tile(tf.expand_dims(self.adj, 0), (tf.shape(self.feats)[0], 1, 1))
        else:
            A_D = self.adj
        wts = tf.tile(tf.expand_dims(weights, 0), (tf.shape(self.feats)[0], 1, 1))
        kernel = tf.matmul(self.feats, wts)
        return tf.matmul(A_D, kernel)

    def __construct_embeddings(self):
        """
        This represents Z_l in the paper. Specifically, the formula: Z_l = GNN_l,embed(A_l,X_l)
        :return: embeds with activation function
        """
        total = []
        for i in range(len(self.embed_weights)):
            total.append(self.__apply_mat_mul(self.embed_weights[i]))
        total_add = tf.add_n(total)
        return self.act(total_add + self.bias)

    def __construct_pool(self):
        """
        Represents the S_l in the paper. (looks similar to the pytorch implementation from the author)
        :return:
        """
        result = self.__apply_mat_mul(self.pool_weights)
        return tf.nn.softmax(result, axis=-1)

    def __call__(self, *args, **kwargs):
        Z = self.__construct_embeddings()
        S = self.__construct_pool()

        X = tf.matmul(tf.transpose(S, perm=[0, 2, 1]), Z)

        if len(self.adj.shape) == 2:
            A_D = tf.tile(tf.expand_dims(self.adj, 0), (tf.shape(self.feats)[0], 1, 1))
        else:
            A_D = self.adj

        A = tf.matmul(tf.matmul(tf.transpose(S, perm=[0, 2, 1]), A_D), S)
        return X, A


class GraphAttentionPooling(object):

    """
    From the paper, https://arxiv.org/pdf/1703.03130.pdf  but with graphs instead of LSTMs:
    """
    def __init__(self, feats, attention, bias_dim, flatten=False):
        self.feats = feats
        self.ws1 = tf.Variable(tf.random_uniform([attention, int(self.feats.shape[-1])]))
        self.ws2 = tf.Variable(tf.random_uniform([bias_dim, attention]))
        self.flatten = flatten

    def __call__(self, *args, **kwargs):
        feats_t = tf.transpose(self.feats, perm=[0,2,1])
        weights = tf.tile(tf.expand_dims(self.ws1, 0), (tf.shape(self.feats)[0], 1, 1))
        t_aux = tf.tanh(tf.matmul(weights, feats_t))
        bias = tf.tile(tf.expand_dims(self.ws2, 0), (tf.shape(self.feats)[0], 1, 1))
        A = tf.nn.softmax(tf.matmul(bias, t_aux))
        out = tf.matmul(A, self.feats)
        if self.flatten:
            return tf.reshape(out, [tf.shape(self.feats)[0], out.get_shape().as_list()[1] * out.get_shape().as_list()[2]])
        return out
