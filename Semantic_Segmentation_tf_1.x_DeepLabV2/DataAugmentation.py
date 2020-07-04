import cv2
import csv
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt

class DataAugmentation:
	def crop_image(self):

		for i in len(range(1, 10)):
			print("Happy")

if __name__ == "__main__":
	print('Image augmentation:')
	print('Preparing training and validation datasets...', end='', flush=True)
	# samples = DataAugmentation.prepare_dataset('data/track_1_forwards.csv')
	# samples = DataAugmentation.prepare_dataset('data/track_2_special.csv')
	#samples = DataAugmentation.prepare_dataset('data/track_1_fbrus_2_fbrs_ra0.9.csv')
	samples = DataAugmentation.prepare_dataset('Bandappa/train_data/temp.txt')
	#/home/kpit/tesnsorflow_projects/deeplabres/Bandappa/train_data
	# samples = DataAugmentation.prepare_dataset('data/track_1_udacity.csv')
	print('done')
'''
    #
    # Show random shadow images
    #
    # for i in range(len(samples)):
    #     idx = randint(0, len(samples))
    #     dataset_path = './data'
    #
    #     # load random images
    #     image = cv2.imread(dataset_path + '/' + samples[idx][0].lstrip())
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = DataAugmentation.random_shadow(image)
    #     cv2.imshow('Shadow augmentation', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(700)

    #
    # Show augmented images
    #
    print('Preparing figures...', end='', flush=True)
    nb_shown_images = 5
    nb_samples = len(samples)

    # blurred, equalized, brightness and shadow images
    fig1, axarr1 = plt.subplots(nb_shown_images, 5, figsize=(16, 9))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05, wspace=0.03, hspace=0.03)

    # flipped, rotated and transformed images
    fig2, axarr2 = plt.subplots(nb_shown_images, 5, figsize=(16, 9))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05, wspace=0.0, hspace=0.19)

    # cropped images
    fig3, axarr3 = plt.subplots(nb_shown_images, 5, figsize=(12, 9))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05, wspace=0.03, hspace=0.03)

    # center, left and right image
    fig4, axarr4 = plt.subplots(nb_shown_images, 3, figsize=(10, 9))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.05, wspace=0.0, hspace=0.19)
'''