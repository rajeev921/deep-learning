import tensorflow as tf 
import numpy as np
import glob
from PIL import Image

def create_bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images = glob.glob('dummy_images/*.png')
labels = glob.glob('dummy_labels/*.png')

writer = tf.io.TFRecordWriter('dataset.tfrecords')

for image,label in zip(images, labels):
    img = Image.open(image)
    lbl = Image.open(label)
    feature = { 'label': create_bytes_feature(np.array(lbl).tostring()), 'image': create_bytes_feature(np.array(img).tostring()) }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()