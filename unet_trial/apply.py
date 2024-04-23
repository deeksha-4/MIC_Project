import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
import unet3
from unet3 import unet
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate, Flatten, Dropout

def load_images(imgsort, masksort, image_dir, mask_dir):
    images, masks = [], []

    for img, msk in tqdm(zip(imgsort, masksort), total = len(imgsort), desc = 'Loading Images and Masks'):
        image = cv2.imread(image_dir + img, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_dir + msk, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256,256))
        mask = cv2.resize(mask, (256,256))

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        images.append(image)
        masks.append(mask)

        del image, mask
        
    return images, masks
def plot_image_with_mask(image_list, mask_list, num_pairs = 4):
    plt.figure(figsize = (18,9))
    for i in range(num_pairs):
        idx = random.randint(0, len(image_list))
        img = image_list[idx]
        mask = mask_list[idx]
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f'Real Image, index = {idx}')
        plt.axis('off')
        plt.subplot(2, 4, i + num_pairs + 1)
        plt.imshow(mask)
        plt.title(f'Segmented Image, index = {idx}')
        plt.axis('off')
        del img, mask

def to_tensor(image_list, mask_list):
    tensor_images = tf.convert_to_tensor(tf.cast(np.array(image_list),  dtype = tf.float32))/255
    tensor_masks = tf.convert_to_tensor(tf.cast(np.array(mask_list), dtype= tf.float32))/255
    return tensor_images, tensor_masks

if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    image_dir = 'dataset/images/'
    mask_dir = 'dataset/masks/'

    imgsort = sorted(os.listdir(image_dir))[1:-1]
    masksort = sorted(os.listdir(mask_dir))
    print(len(imgsort), len(masksort))
    images, masks = load_images(imgsort, masksort, image_dir, mask_dir)
    print(len(images), len(masks))
    plot_image_with_mask(images, masks, num_pairs = 4)
    images, masks = to_tensor(images, masks)
    train_split = tf.cast(tf.round(len(images)*0.6) - 1, dtype = tf.int32)
    test_val_split = tf.cast(tf.round(len(images)*0.2), dtype = tf.int32)

    images_train = images[:train_split]
    masks_train = masks[:train_split]

    images_val = images[train_split:train_split + test_val_split]
    masks_val = masks[train_split:train_split + test_val_split]

    images_test = images[train_split + test_val_split:]
    masks_test = masks[train_split + test_val_split:]

    del images, masks

    print(f'The length of images and masks for training is {len(images_train)} and {len(masks_train)} respectively')
    print(f'The length of images and masks for validation is {len(images_val)} and {len(masks_val)} respectively')
    print(f'The length of images and masks for testing is {len(images_test)} and {len(masks_test)} respectively')