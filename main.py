#! /usr/bin/env python
# This code is modified from the original by Denny Britz
# found here: github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
import os
import time
import datetime
from sklearn import metrics
import corpus_utils
from CNN import TextCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every",20 , "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 40, "Save model after this many steps")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("dev_parameter_search", False, "Run experiment only on dev set")
tf.flags.DEFINE_string("checkpoint_dir","", "Checkpoint to resume")
tf.flags.DEFINE_boolean("test", False, "Allow device soft device placement")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size

# Output directory for models and summaries
timestamp = str(int(time.time()))
if FLAGS.checkpoint_dir:
    out_dir = os.path.split(FLAGS.checkpoint_dir)[0]
else:
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
print("Writing to {}\n".format(out_dir))
parameter_file = os.path.join(out_dir, "parameters.txt")

print("\nParameters:")
with open(parameter_file,'wb') as fo:
    for attr, value in sorted(FLAGS.__flags.iteritems()):
        print("{}={}".format(attr.upper(), value))
        fo.write("{}={}\n".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================
n_authors = 10
# Load data
print("Loading data...")
x, y = corpus_utils.loadData(n_authors)
vocabulary = corpus_utils.loadVocab(n_authors)
# Randomly shuffle data
x_splits,y_splits = corpus_utils.traindevtestSplit(x,y)
x_train, x_dev, x_test = x_splits
y_train, y_dev, y_test = y_splits

print("Vocabulary Size: {:d}".format(len(vocabulary)+1))
print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev),len(y_test)))



# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocabulary)+1,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)



        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        eval_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        
        # Test summaries
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.train.SummaryWriter(test_summary_dir, sess.graph_def)

        # restore from checkpoint?
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Otherwise initialize all variables
            sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def eval_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a non-training set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0 # don't perform dropout on evaluation
            }
            step, summaries, loss, accuracy = sess.run([global_step, eval_summary_op, cnn.loss, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        if FLAGS.dev_parameter_search:
            x_y = zip(x_dev, y_dev)
        else:
            x_y = zip(x_train, y_train)
        
        if not FLAGS.test:
            batches = corpus_utils.batch_iter(x_y, FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    if FLAGS.dev_parameter_search:
                        eval_step(x_dev[:300], y_dev[:300], writer=dev_summary_writer)
                    else:
                        eval_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            # Test model once
        else:
            pred = sess.run(cnn.predictions, {cnn.input_x: x_test,cnn.dropout_keep_prob: 1.0}) 
            f1 = metrics.f1_score(corpus_utils.unOneHot(y_test),pred,average='weighted')
            acc = metrics.accuracy_score(corpus_utils.unOneHot(y_test),pred)
            print "F1 score: ", f1
            print "Accuracy: ", acc