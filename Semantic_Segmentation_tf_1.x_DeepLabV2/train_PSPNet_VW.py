"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

#from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from kaffe.tensorflow import Network

from deeplab_resnet import ImageReader, decode_labels, inv_preprocess, prepare_label
from deeplab_resnet.model_VW_473 import DeepLabResNetModel,bottom_net

IMG_MEAN = np.array((81.9495818602, 84.3999660303, 80.4862847463), dtype=np.float32)#BGR

#IMG_MEAN = np.array((128.0,128.0,128.0), dtype=np.float32)

BATCH_SIZE = 3
DATA_DIRECTORY = '/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/train_data/'
DATA_LIST_PATH = '/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/train_data/temp.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '473,473'
LEARNING_RATE = 0.0000106928 #0.1 #0001
MOMENTUM = 0.9
NUM_CLASSES = 53
NUM_STEPS = 50000
POWER = 0.9
RANDOM_SEED = 160
RESTORE_FROM = '/home/kpit/tesnsorflow_projects/deeplabres/models/deeplab_resnet.ckpt'
SAVE_NUM_IMAGES = 3
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshot_noMulti_0.52_473_VW_croppedData/'
WEIGHT_DECAY = 0.0005


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.
    output_0=net.layers['res5c_relu']
    output_0=tf.image.resize_bilinear(output_0,[60,60])
    output_1 = net.layers['conv5_3_pool1_conv/bn_relu']
    output_1=tf.image.resize_bilinear(output_1,[60,60])
    output_2 = net.layers['conv5_3_pool2_conv/bn_relu']
    output_2=tf.image.resize_bilinear(output_2,[60,60])
    output_3 = net.layers['conv5_3_pool3_conv/bn_relu']
    output_3=tf.image.resize_bilinear(output_3,[60,60])
    output_6 = net.layers['conv5_3_pool6_conv/bn_relu']
    output_6=tf.image.resize_bilinear(output_6,[60,60])

    #conv5_3_concat_1_90_4096=tf.concat([output_1,output_2,output_3,output_6,output_0],3,name='conv5_3_concat_1_90_4096')
    

    output_cat_combine = tf.concat([output_1,output_2,output_3,output_6,output_0],3,name='output_cat_combine')

    
    net_bottom = bottom_net({'conv5_3_concat':output_cat_combine},is_training=args.is_training, num_classes=args.num_classes)
    output_fin=net_bottom.layers['conv6']
    raw_output=tf.image.resize_bilinear(output_fin,[473,473])
    
    restore_var = [v for v in tf.global_variables()]
    #else:
    #restore_var = [v for v in tf.global_variables()  if 'conv6' not in v.name]
    #restore_var = [v for v in tf.global_variables()  if ('conv5' not in v.name and 'conv6' not in v.name and 'res3b3_relu_pool' not in v.name and 'res3b3_concat_90_2560_conv_512' not in v.name)]
    #for f in restore_var:
    #	print(f.name)
    #exit(0) 
    #restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables() if ('conv/bn_relu' in v.name) or ('beta' not in v.name and 'gamma' not in v.name)]
    fc_trainable = [v for v in all_trainable if ('conv5' in v.name or 'conv6' in v.name) and ('conv/bn_relu' not in v.name) ]
    batchNorm_trainable = [v for v in all_trainable if  'conv/bn_relu'  in v.name ]
    conv_trainable = [v for v in all_trainable if ('conv5' not in v.name and 'conv6' not in v.name)] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    #fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    #print len(fc_b_trainable)
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable)) + len(batchNorm_trainable)
    assert(len(fc_trainable) == len(fc_w_trainable))
    #train_var = [v for v in tf.global_variables()  if ('conv5'  in v.name) or ('conv6'  in v.name) or ('res3b3_relu_pool' in v.name)]
    train_var =   [v for v in tf.trainable_variables()]  

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)


    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)

    total_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                          graph=tf.get_default_graph())

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    grad_op = tf.train.MomentumOptimizer(learning_rate, args.momentum)#tf.train.GradientDescentOptimizer(learning_rate)
    train_op = grad_op.minimize(reduced_loss,var_list=train_var)

            

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    
    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        #load(loader, sess, args.restore_from)
        print 'restoring from previous snapshot' 
        loader.restore(sess,tf.train.latest_checkpoint(SNAPSHOT_DIR))
    #exit(0)
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        
        if step % args.save_pred_every == 0 and step != 0:
            #loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, train_op], feed_dict=feed_dict)
            loss_value, images, labels, preds, aa,summary = sess.run([reduced_loss, image_batch, label_batch, pred, train_op,total_summary], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            # save(saver, sess, args.snapshot_dir+'model.ckpt')
            saver.save(sess,args.snapshot_dir+ "model.ckpt")
            #test_writer.add_summary(loss_summary_,step)


        #else:
        loss_value, aa,im,lb,lr = sess.run([reduced_loss, train_op,image_batch, label_batch,learning_rate], feed_dict=feed_dict)
        #im,lb = sess.run([image_batch, label_batch], feed_dict=feed_dict)
        #print im.shape,lb.shape
        duration = time.time() - start_time
        print "step=",step,"loss=",loss_value,"duration=",duration,"lr_rate=",lr
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
