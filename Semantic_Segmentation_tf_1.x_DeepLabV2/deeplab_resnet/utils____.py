from PIL import Image
import numpy as np
import tensorflow as tf

# colour map
label_colours1 = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

label_colours=[(128, 128, 0), (255, 0, 255), (180, 50, 180), (180, 50, 100), (50, 50, 100), (128, 0, 255),
              (128, 80, 255), (200, 125, 210), (150, 0, 150), (210, 50, 115), (147, 253, 100), (139, 99, 108),
              (150, 150, 200), (150, 70, 150), (150, 255, 150), (180, 150, 200), (180, 50, 150),
              (255, 150, 255), (135, 206, 255), (238, 233, 191), (110, 110, 0), (204, 255, 153), (182, 89 ,6),
              (150, 50, 4), (90, 30, 1), (204, 0, 204), (122, 0, 122), (100, 0, 100), (90,18,185), (68, 20, 100),
              (204, 165, 230), (255, 0, 0), (200, 0, 0), (150, 0, 0), (100, 0, 0), (64, 128, 128), (128, 60, 60), 
              (250, 126, 62), (72, 209, 204), (255, 128, 128), (204, 153, 255), (189, 73, 155), (239, 89, 191), 
              (204, 153, 200), (0, 255, 0), (0, 200, 0), (0, 150, 0), (0, 0, 100), (255, 128, 0), (200, 128, 0), 
              (150, 128, 0), (150, 128, 51), (0, 65, 130), (0, 17, 35), (255, 255, 0), (255, 255, 200), (241, 230, 255), 
              (255, 70, 185), (238, 162, 173), (64, 0, 64), (147, 253, 194), (255, 0, 128), (255, 246, 143), (133, 117, 71), 
              (233, 100, 0), (185, 122, 87), (159, 121, 238), (100, 255, 100), (180, 80, 100), (220, 90, 100), (255, 100, 100),
              (0, 128, 255), (0, 255, 255), (0, 200, 100)]

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def decode_labels_to_trainID(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('L', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = k
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs
