import tensorflow as tf
import numpy as np
from gcn.layers import GCNLayer
import pandas as pd
from gcn.expression_features import CorrelationExpressionGraph
from sklearn.model_selection import train_test_split
import argparse
import json

tf.logging.set_verbosity(tf.logging.INFO)


class TensorflowExpressionGraph(object):

    def __init__(self):
        self.model = tf.estimator.Estimator(TensorflowExpressionGraph.build_model)

    @staticmethod
    def build_graph(x_dict, training):
        gcn_layer1 = GCNLayer(x_dict['features'], 35,
                              activation=tf.nn.relu, support=1,
                              kernel_initializer=tf.glorot_uniform_initializer(seed=12345))()
        do1 = tf.layers.dropout(gcn_layer1, 0.25, training=training)
        gcn_layer2 = GCNLayer(do1, 35,
                              activation=tf.nn.relu, support=1,
                              kernel_initializer=tf.glorot_uniform_initializer(seed=12345))()
        do2 = tf.layers.dropout(gcn_layer2, 0.25, training=training)
        flattened = tf.layers.flatten(do2)
        return tf.layers.dense(flattened, 2, kernel_initializer=tf.glorot_uniform_initializer(seed=12345))

    @staticmethod
    def build_model(features, labels, mode):
        logits = TensorflowExpressionGraph.build_graph(features, mode == tf.estimator.ModeKeys.TRAIN)

        if mode == tf.estimator.ModeKeys.PREDICT:
            inner_preds = tf.argmax(logits, 1, name='out')
            export_outputs = {
                "serving_default": tf.estimator.export.PredictOutput(inner_preds)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=inner_preds, export_outputs=export_outputs)
        loss_op = tf.reduce_sum(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
        log_hook = tf.train.LoggingTensorHook({"loss": loss_op}, every_n_iter=10, )
        metric_op = tf.metrics.accuracy(labels=labels, predictions=logits)

        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=logits,
            training_hooks=[log_hook],
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'mse': metric_op})

        return estim_specs

    def train_model(self, x, y, epochs=2000, batch_size=256):
        """
        :param x: Features
        :param y: labels
        :param epochs: number of training iterations
        """
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'features': x}, y=y,
            batch_size=batch_size, num_epochs=None, shuffle=True)
        self.model.train(input_fn, steps=epochs)

    def save_model(self, path):
        self.model.export_savedmodel(export_dir_base=path,
                                     serving_input_receiver_fn=TensorflowExpressionGraph.serving_input_fn)

    def predictions(self, x):
        x = x.astype(np.float32)
        input_fn = tf.estimator.inputs.numpy_input_fn(x={'features': x}, shuffle=False)
        return self.model.predict(input_fn)

    def full_predictions(self, x):
        preds = self.predictions(x)
        all_d = []
        for pred in preds:
            all_d.append(pred)
        return np.asarray(all_d)

    def get_model(self):
        return self.model

    @staticmethod
    def serving_input_fn():
        tensor = {'features': tf.placeholder(tf.float32, shape=[None, None, 1])}
        return tf.estimator.export.ServingInputReceiver(tensor, tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple arguments for estimator.')
    parser.add_argument('--file', dest='file_name', default='example.csv',
                        help='file name for running')
    parser.add_argument('--label', dest='label', default='risk',
                        help='label in file')
    parser.add_argument('--ignore-columns', dest='ignore_columns', default=['sample_id'], nargs='+',
                        help='ignore_columns')
    parser.add_argument('--epochs', dest='epochs', default=4000, type=int,
                        help='epochs')
    parser.add_argument('--replace-labels', dest='replace_labels', default='{"high": 1.0, "low": 0.0}',
                        help='replace labels json object')
    args = parser.parse_args()
    replace_labels = json.loads(args.replace_labels)

    df = pd.read_csv(args.file_name)

    expression_builder = CorrelationExpressionGraph(df, label_column=args.label, ignore_columns=args.ignore_columns)
    feats, labs = expression_builder.build_features_apply_mult(replace_labels=replace_labels)
    X_train, X_test, y_train, y_test = train_test_split(feats, labs, test_size=0.15, random_state=12345)

    tgraph = TensorflowExpressionGraph()
    tgraph.train_model(X_train, y_train, epochs=args.epochs, batch_size=300)
    print("Test Error: %s" % str(np.abs(np.argmax(y_test, axis=1) - tgraph.full_predictions(X_test)).sum()))
