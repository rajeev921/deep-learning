#!/usr/bin/python 
import os, sys
import PIL
from PIL import Image
import numpy as np
import cv2
import glob
'''
DATA_DIRECTORY = '/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/1280_800_trainData/5_1_18/'
SAVE_DIRECTORY = '/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/1280_800_trainData/new/'


dirs = os.listdir(image_path)

def resize():
	for item in dirs:
		if os.path.isfile(path+item):
		    im = Image.open(path+item)
		    print("path + item", path , "   ", item)
		    f, e = os.path.splitext(path+item)
		    print(f, "   ", e)
		    imResize = im.resize((200, 200), Image.ANTIALIAS)
		    imResize.save(f + '/new' + ' resized.jpg', 'JPEG', quality=90)

resize()

'''
img = cv2.imread('/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/1280_800_trainData/5_1_18/Original/front_1502264283435766.png')
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)


