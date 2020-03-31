# First download CIFAR-10 dataset from:
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Then unzip the package
# Finally replace the glob search pattern in line 13 and run the script.

import glob
import pickle

import cv2
import tensorflow as tf

objects = sorted(
    glob.glob('/home/work/cifar-10-python/cifar-10-batches/data_*'))
if not objects:
    raise RuntimeError(
        'No files found. Make sure the search pattern is correct.')

for i, data_batch in enumerate(objects):
    tfrecord_name = 'cifar10-{}.tfrecord'.format(i)
    writer = tf.io.TFRecordWriter(tfrecord_name)
    with open(data_batch, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    all_data = d[b'data']
    all_label = d[b'labels']
    assert all_data.shape[0] == len(all_label)

    for j in range(all_data.shape[0]):
        label = all_label[j]
        data = cv2.imencode('.jpg', all_data[j, :].reshape(3, 32, 32).
                            transpose(1, 2, 0))[1].tobytes()
        label_feat = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label]))
        data_feat = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[data]))

        feat_dict = {
            'data': data_feat,
            'label': label_feat
        }

        example = tf.train.Example(
            features=tf.train.Features(feature=feat_dict))
        writer.write(example.SerializeToString())
    writer.close()
