import tensorflow as tf
from gcn.layers import GraphDiffPool, GCNLayer
import pandas as pd
from gcn.expression_features import CorrelationExpressionGraph
from sklearn.model_selection import train_test_split
import argparse
import json
import numpy as np

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
    feats, labs, A = expression_builder.build_features_with_adjacency(replace_labels=replace_labels)
    X_train, X_test, y_train, y_test = train_test_split(feats, labs, test_size=0.15, random_state=12345)

    features = tf.placeholder(tf.float32, shape=(None, 253, 1))
    adjacency = tf.placeholder(tf.float32, shape=(253, 253))
    labels = tf.placeholder(tf.float32, shape=(None, 2))

    gcn_layer1, A_1 = GraphDiffPool(features, 32, adjacency, pool_size=16, kernel_initializer=tf.glorot_uniform_initializer(12345))()
    gcn_layer2 = GCNLayer(gcn_layer1, 32, A_1, kernel_initializer=tf.glorot_uniform_initializer(12345))()
    flatten = tf.layers.flatten(gcn_layer2)
    out = tf.layers.dense(flatten, 2, kernel_initializer=tf.glorot_uniform_initializer(seed=12345))

    loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=out))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss_op)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, 4000):
            x = sess.run([out, train_op, loss_op], feed_dict={features: X_train, adjacency: A, labels: y_train})
            if i % 20 == 0:
                print("Loss: %s" % (str(x[2])))
                print(np.abs(np.argmax(y_train, axis=1) - np.argmax(x[0], axis=1)).sum())