import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as kbe
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import pai

tf.app.flags.DEFINE_string("input_pattern", "", "Input tfrecord data pattern")
tf.app.flags.DEFINE_string("model_dir", "./output", "Directory for export model")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.app.flags.DEFINE_integer("max_steps", 2000, "max training steps")
tf.app.flags.DEFINE_integer("gpu_num", 4, "number of gpus")
tf.app.flags.DEFINE_integer("save_summary_steps", 1000, "save summary per steps")
tf.app.flags.DEFINE_integer("log_step_count_steps", 100, "log per steps")

FLAGS = tf.flags.FLAGS

def create_model():
    model = Sequential(name='CIFAR-10')

    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                      padding='same', kernel_initializer='he_uniform',
                      kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10))

    return model

def get_files_slice(files):
  import json
  import os
  tf_config = os.environ.get("TF_CONFIG")
  if not tf_config:
    return files
  tf_config_json = json.loads(tf_config)
  task_type = tf_config_json.get("task", {}).get("type")
  chief = tf_config_json.get("cluster",{}).get("chief")
  worker = tf_config_json.get("cluster",{}).get("worker")

  num_chief = 0
  num_worker = 0
  if chief:
    num_chief = len(chief)
  if worker:
    num_worker = len(worker)
  num_total = num_chief + num_worker

  if task_type == "chief":
    task_index = tf_config_json.get("task", {}).get("index")
  else:
    task_index = tf_config_json.get("task", {}).get("index") + num_chief

  files.sort()
  files_slice = [f for i, f in enumerate(files) if i % num_total == task_index]
  print("number of workers:", num_total,
        " task_index:", task_index,
        " files slice:", files_slice)
  return files_slice


def get_dataset(ds_pattern):
    files = tf.io.gfile.glob(ds_pattern)
    files = get_files_slice(files)
    if not files:
        raise RuntimeError("Could not find any files: {}".format(ds_pattern))
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=4)
    return ds


def parse_dataset(ds, batch_size=32):
    feature_discription = {
        'label': tf.io.FixedLenFeature([1], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)
    }
    def decode(example):
        feature = tf.io.parse_single_example(example, feature_discription)
        data = tf.image.decode_jpeg(feature['data'])
        data = tf.image.convert_image_dtype(data, tf.float32)
        label = feature['label']
        label = kbe.squeeze(label, axis=0)
        return data, label

    return ds.map(decode, num_parallel_calls=8)      \
             .cache()                                \
             .shuffle(50000)                         \
             .batch(batch_size, drop_remainder=True) \
             .prefetch(batch_size*3)                 \
             .repeat(-1)


def get_sgd_optimizer():
    return tf.train.GradientDescentOptimizer(learning_rate=0.001)


def get_available_gpus():
    gpus = ["/device:GPU:"+str(i) for i in range(FLAGS.gpu_num)]
    print("gen gpu devices:", gpus)
    return gpus


def get_train_strategy():
    try:
        # use MirroredStrategy
        #devices = get_available_gpus()
        #strategy = tf.distribute.MirroredStrategy(devices=devices)

        # use ExascaleStrategy
        strategy = pai.distribute.ExascaleStrategy(
                                                   max_splits=1,
                                                   default_device='/cpu:0',
                                                   num_gpus=FLAGS.gpu_num)
        return strategy
    except Exception as e:
        print(str(e))
        sys.exit(-1)


def train_input_fn():
    dataset = parse_dataset(get_dataset(FLAGS.input_pattern),
                            batch_size=FLAGS.batch_size)
    return dataset


def model_fn(features, labels, mode):
    model = create_model()
    logits = model(features, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      opt = get_sgd_optimizer()
      train_op = opt.minimize(loss, global_step=global_step, name='train')
      return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metric_ops={'no_eval': (tf.no_op(), tf.no_op())})
    else:
      raise ValueError(
          "Only TRAIN and EVAL modes are supported: %s" % (mode))


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.random.set_random_seed(12345)
    # Below is the strategy definition.
    distribute = get_train_strategy()
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        config=tf.estimator.RunConfig(
            train_distribute=distribute,
            session_config=session_config,
            save_summary_steps=FLAGS.save_summary_steps,
            log_step_count_steps=FLAGS.log_step_count_steps))

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=FLAGS.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:None)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print('===End of program.===')


if __name__ == '__main__':
    main()
