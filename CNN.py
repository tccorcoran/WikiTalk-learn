# This code is modified from the original by Denny Britz
# found here: github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                W1 = tf.Variable(tf.truncated_normal([filter_size, embedding_size, 1, num_filters], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
                
                W2 = tf.Variable(tf.truncated_normal([1, 1, num_filters, 2*num_filters], stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[2*num_filters]), name="b2")
                
                # 1st Convolution Layer
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b1), name="relu")
                # Maxpooling over the outputs
                
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                
                # 2nd Convolution Layer
                #set_trace()
                conv = tf.nn.conv2d(pooled, W2, strides=[1, 1, 1, 1],padding="VALID", name="conv2")
                
                # non-linear activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b2), name="relu")
                # Max pooling
                
                pooled = tf.nn.max_pool(h,ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool2")

                pooled_outputs.append(pooled)
                
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, 2*num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) logits and predictions
        with tf.name_scope("model"):
            W = tf.Variable(tf.truncated_normal([2*num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

