"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os,glob
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np
import cv2

from deeplab_resnet import ImageReader, decode_labels, inv_preprocess, prepare_label
#from deeplab_resnet.model_VW_713 import DeepLabResNetModel,bottom_net
from deeplab_resnet.model_VW_473 import DeepLabResNetModel,bottom_net

IMG_MEAN = np.array((73.00698793,83.66876762,72.67891434), dtype=np.float32)
    
NUM_CLASSES = 53
#SAVE_DIR = './output_vw_713/'
SAVE_DIR = '/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/inference/output/20170821_Wolle_StadtWOB_A39_1_of_1/'
SNAPSHOTS = './snapshot_noMulti_0.52_473_VW_croppedData/'
image_test_directory = '/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/inference/input/20170821_Wolle_StadtWOB_A39_1_of_1/'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    #parser.add_argument("--video-path", type=str,default=VIDEOPATH,
    #                    help="Path to the RGB image file.")
    parser.add_argument("--model-weights", type=str,default=SNAPSHOTS,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    #saver = tf.train.import_meta_graph('snapshot/model.ckpt-4.meta')
    #saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    #print("Restored model parameters from {}".format(ckpt_path))\

def input_gen(img_path):
    #global frame_index
    # keep looping infinitely

    # source: http://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture
    #cap = cv2.VideoCapture(video_path)
    #cap.set(1, frame_index)

    # read the next frame from the file, Note that frame is returned as a Mat.
    # So we need to convert that into a tensor.
    #(grabbed, frame) = cap.read()
    #if not grabbed:
            #coord.request_stop()
        #    yield frame_index,None
    frame = cv2.imread(img_path)
    img = np.asarray(frame,dtype=np.float32)
    #frame_index += 1
    #to_retun = deform_images(img)
    #img = cv2.resize(img,(713,713),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
    img = cv2.resize(img,(473,473),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
    #img = np.transpose(img,(2,0,1))
    #img -= mean
    
    #img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    #img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    #img -= IMG_MEAN 
    #imgBGR = np.transpose(img,(2,0,1))
    #print(imgBGR.shape)
    #imgBGR = imgBGR - IMG_MEAN[:, np.newaxis, np.newaxis]
    #imgBGR = np.transpose(img,(1,0,2))
    print(img.shape)
    #return img
    yield img


def main1():
    args = get_arguments()
    global frame_index
    a=input_gen(args.video_path)
    for img in next(a):
        print(img.shape)
        print (frame_index)

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    #img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=3)
    #img = tf.placeholder(tf.float32,[713,713,3])
    img = tf.placeholder(tf.float32,[473,473,3])
    #img= tf.image.resize_images(img, [713, 713])

    # Convert RGB to BGR.
    img_b, img_g, img_r = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN 
    #exit()
    
    # Create network.
    #frame_index = 0
    #print (frame_index)
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    
    # Predictions.
    #raw_output = net.layers['fc1_voc12']
    #raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2,])
    output_0=net.layers['res5c_relu']
    output_1 = net.layers['conv5_3_pool1_conv/bn_relu']
    output_1=tf.image.resize_bilinear(output_1,[60,60])
    #output_1=tf.image.resize_bilinear(output_1,[90,90])   for 713 * 713 image
    output_2 = net.layers['conv5_3_pool2_conv/bn_relu']
    output_2=tf.image.resize_bilinear(output_2,[60,60])
    #output_2=tf.image.resize_bilinear(output_2,[90,90])     for 713 * 713 image
    output_3 = net.layers['conv5_3_pool3_conv/bn_relu']
    output_3=tf.image.resize_bilinear(output_3,[60,60])
    #output_3=tf.image.resize_bilinear(output_3,[90,90])     for 713 * 713 image
    output_6 = net.layers['conv5_3_pool6_conv/bn_relu']
    output_6=tf.image.resize_bilinear(output_6,[60,60])
    #output_6=tf.image.resize_bilinear(output_6,[90,90])     for 713 * 713 image
    
    output_cat_combine=tf.concat([output_1,output_2,output_3,output_6,output_0],3,name='conv5_3_concat_1')
    #filter_=tf.Variable(tf.random_normal[1,3,3,4096])
    #ot=tf.nn.conv2d(output_cat_combine,filter_,strides=[1,1,1,1])
    #print ot.shape
    #exit()

    net_bottom = bottom_net({'conv5_3_concat':output_cat_combine},is_training=False, num_classes=args.num_classes)
    output_fin=net_bottom.layers['conv6']
    #raw_output_up=tf.image.resize_bilinear(output_fin,[713,713])
    raw_output_up=tf.image.resize_bilinear(output_fin,[473, 473])
    restore_var = tf.global_variables()

    
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    #load(loader, sess, args.model_weights)
    loader.restore(sess,tf.train.latest_checkpoint(args.model_weights))
    
    # Perform inference.
    
    #for frame_index,im in next(frame_grabber):
    #cap = cv2.VideoCapture(args.video_path)
    #fourcc=cv2.VideoWriter_fourcc(*'H264')
    #vid_writer = cv2.VideoWriter(args.save_dir +'pspnet_713_tf.mp4',fourcc,10.0,(713*2,713))
    print ('outside')
    print(image_test_directory)
    #for f in glob.glob(image_test_directory+"/*.bmp"):
    for f in glob.glob(image_test_directory+"/*.bmp"):
        #(grabbed, frame) = cap.read()
        #if not grabbed:
        #   break
        #frame_index+=1
        print ('inside')
        frame = cv2.imread(f)
        print (frame.shape,type(frame),)
        frame = cv2.resize(frame,(473,473),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
        #frame = cv2.resize(frame,(713,713),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
        feed_dict = { img : frame }
        preds = sess.run(pred,feed_dict=feed_dict)

        msk = decode_labels(preds, num_classes=args.num_classes)
        #im = Image.fromarray(msk[0])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        #im.save(args.save_dir + 'mask'+str(frame_index)+'.png')
        #im = np.asarray(im)
        
        final = np.zeros((473,473*2,3),dtype=np.uint8)
        #final = np.zeros((713,713*2,3),dtype=np.uint8)
        msk_bgr = cv2.cvtColor(msk[0],cv2.COLOR_BGR2RGB)
        final = np.concatenate((frame, msk_bgr), axis=1)
	
        cv2.imshow("a",msk_bgr)
        cv2.imwrite(SAVE_DIR+f.split('/')[-1],msk[0])
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
            #vid_writer.write(final)
            print('The output file has been saved to {}'.format(args.save_dir + 'mask'+str(frame_index)+'.png'))

        #cap.release()
        #vid_writer.release() 
if __name__ == '__main__':
    main()
