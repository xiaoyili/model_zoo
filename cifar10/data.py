import tensorflow as tf
from tensorflow.python.keras import backend as kbe

feature_discription = {
    'label': tf.io.FixedLenFeature([1], tf.int64),
    'data': tf.io.FixedLenFeature([], tf.string)
}


def get_dataset(ds_pattern):
    files = tf.io.gfile.glob(ds_pattern)
    if not files:
        raise RuntimeError("Could not find any files: {}".format(ds_pattern))
    ds = tf.data.TFRecordDataset(files)
    return ds


def parse_dataset(ds, batch_size=32):
    def decode(example):
        feature = tf.io.parse_single_example(example, feature_discription)
        data = tf.image.decode_jpeg(feature['data'])
        data = tf.image.convert_image_dtype(data, tf.float32)
        label = feature['label']
        label = kbe.one_hot(label, num_classes=10)
        label = kbe.squeeze(label, axis=0)
        return data, label

    return ds.map(decode).shuffle(50000).batch(batch_size).prefetch(2)
