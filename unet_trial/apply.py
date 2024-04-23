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
        plt.pause(4)
        del img, mask

def to_tensor(image_list, mask_list):
    tensor_images = tf.convert_to_tensor(tf.cast(np.array(image_list),  dtype = tf.float32))/255
    tensor_masks = tf.convert_to_tensor(tf.cast(np.array(mask_list), dtype= tf.float32))/255
    return tensor_images, tensor_masks

def dice_coeff(y_true, y_pred, smooth=1):
    tensor_rank_t = tf.rank(y_true)
    tensor_rank_p = tf.rank(y_pred)

    # Check if the rank is 4
    is_four_dimensions_p = tf.equal(tensor_rank_p, 4)
    is_four_dimensions_t = tf.equal(tensor_rank_t,4)
    # If you're in a TensorFlow session, you can evaluate the result
    with tf.compat.v1.Session() as sess:
        is_four_dimensions_value_p = sess.run(is_four_dimensions_p)
    with tf.compat.v1.Session() as sess:
        is_four_dimensions_value_t = sess.run(is_four_dimensions_t)
    if is_four_dimensions_value_t:
        y_true = tf.squeeze(y_true, axis=-1)  # Squeeze the last dimension if it exists
    if is_four_dimensions_value_p:
        y_pred = tf.squeeze(y_pred, axis=-1)  # Squeeze the last dimension if it exists
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])  # Compute intersection
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])  # Compute union

    dice_coefficient = (2 * intersection + smooth) / (union + smooth)
    return dice_coefficient
# Setting dice coefficient to evaluate our model
# def dice_coeff(y_true, y_pred, smooth = 1):
#     intersection = tf.reduce_sum(y_true*y_pred, axis = -1)
#     union = tf.reduce_sum(y_true, axis = -1) + tf.reduce_sum(y_pred, axis = -1)
#     dice_coeff = (2*intersection+smooth) / (union + smooth)
#     return dice_coeff
#Function to plot the predictions with orginal image, original mask and predicted mask
def plot_preds(idx):
    plt.figure(figsize = (15, 15))
    test_img = images_test[idx]
    test_img = tf.expand_dims(test_img, axis = 0)
    test_img = tf.expand_dims(test_img, axis = -1)
    pred = unet.predict(test_img)
    pred = pred.squeeze()
    thresh = pred > 0.5
    plt.subplot(1,3,1)
    plt.imshow(images_test[idx])
    plt.title(f'Original Image {idx}')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(masks_test[idx])
    plt.title('Actual Mask')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(thresh)
    plt.title('Predicted Mask')
    plt.axis('off')
    
if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    image_dir = 'dataset/images/'
    mask_dir = 'dataset/masks/'

    imgsort = sorted(os.listdir(image_dir))[1:-1]
    masksort = sorted(os.listdir(mask_dir))
    print(len(imgsort), len(masksort))
    images, masks = load_images(imgsort, masksort, image_dir, mask_dir)
    print(len(images), len(masks))
    # plot_image_with_mask(images, masks, num_pairs = 4)
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
    # plot_image_with_mask(images_train, masks_train)
    #Converting the list of tensors into batches to efficiently train the model, computation-wise
    batch_size = 32

    train_data = tf.data.Dataset.from_tensor_slices((images_train, masks_train))
    train_data = train_data.batch(batch_size)

    val_data = tf.data.Dataset.from_tensor_slices((images_val, masks_val))
    val_data = val_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((images_test, masks_test))
    test_data = test_data.batch(batch_size)
    with strategy.scope():
        unet = unet()
        unet.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy', dice_coeff])
    unet.summary()
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
    unet_history = unet.fit(train_data, validation_data = [val_data], epochs = 50, callbacks = [early_stopping])

    #Plotting the loss and accuracy during training and validation
    plt.figure(figsize = (18, 9))
    plt.subplot(1,3,1)
    plt.plot(unet_history.history['loss'])
    plt.plot(unet_history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.subplot(1,3,2)
    plt.plot(unet_history.history['accuracy'])
    plt.plot(unet_history.history['val_accuracy'])
    plt.title('Training vs Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.subplot(1,3,3)
    plt.plot(unet_history.history['dice_coeff'])
    plt.plot(unet_history.history['val_dice_coeff'])
    plt.title('Training vs Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend(['Training Dice Coefficient', 'Validation Coefficient'])
    #evaluating the model, we got 89.54% accuracy. Pretty Good!
    unet.evaluate(test_data)    
    for i in [random.randint(0, 2000) for i in range(10)]:
        plot_preds(i)