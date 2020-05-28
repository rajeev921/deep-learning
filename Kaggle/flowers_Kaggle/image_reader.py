import math, re, os
import numpy as np
import tensorflow as tf 
#from matplotlib import pyplot as plt
AUTO = tf.data.experimental.AUTOTUNE
######################################  Settings ######################################
BATCH_SIZE = 16
#####
IMAGE_SIZE = [192, 192] # At theis size GPU will run out of memory, Try using TPU
# For GPU training, please select 224 x 224 px image size.

DATA_PATH_SELECT = {  # available image sizes
    192: 'data/tfrecords-jpeg-192x192',
    224: 'data/tfrecords-jpeg-224x224',
    331: 'data/tfrecords-jpeg-331x331',
    512: 'data/tfrecords-jpeg-512x512'
}

DATA_PATH = DATA_PATH_SELECT[IMAGE_SIZE[0]]
print(DATA_PATH)
TRAINING_FILENAMES = tf.io.gfile.glob(DATA_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(DATA_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(DATA_PATH + '/test/*.tfrec')

######################################  Datasets ######################################
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0 # Converts image to floats in [0, 1]
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image
    
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64)# shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # return a dataset of (image, label) pairs
    
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string)# shape [] means single element
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)
    
def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords.  Fot optimal performanc, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway. 
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically reads from multiple files
    dataset = dataset.with_options(ignore_order) # Use data as soon as it streams in, than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    
    #returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)