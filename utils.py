import tensorflow as tf
import numpy as np
import h5py, os, pdb
from tensorflow.keras import backend as K

def conv_mask_gt(z):
   # Given a mask, returns two class tensor for every pixel
    z = tf.convert_to_tensor(z)

    background = tf.cast(( z == 0), dtype = 'float32')
    manipulated = tf.cast(( z == 1), dtype = 'float32')


    return [background, manipulated] 

def transform_masks(mask):
    result = np.zeros((*mask.shape, 2))
    result[:, :, 0] = 1 - mask
    result[:, :, 1] = mask
    return result

def transform_predictions(pred, threshold):
    m_pixels = pred[:, :, 1]
    return np.where(m_pixels >= threshold, 1, 0) 

def validate_input_images(hdf5_file, images_label, masks_label):
    if not os.path.exists(hdf5_file):
        print("Fail. ", hdf5_file, " doesn't exist.")
        return False
    
    f = h5py.File(hdf5_file, 'r')

    if images_label not in f or masks_label not in f:
        print("Fail. ", hdf5_file, " doesn't contain ", images_label, " and ", masks_label, " datasets.")
        return False

    images = f[images_label]
    masks = f[masks_label]

    # Validate if num of images and masks are the same
    if images.shape[0] != masks.shape[0]:
        print("Fail. Amount of images and masks don't match.")    
        return False

    # Validate that groundtruth only contain 0 and 1's
    unique_values = np.unique(masks)
    one_zero_array = np.array( [0,1] ).astype(np.float64)
    contains_only_one_zero = np.array_equal(unique_values, one_zero_array)

    if not contains_only_one_zero:
        print("Fail. Grountruth contain values different to 0's and 1's.")
        return False

    return True

# --------------------------------------------------------------------------------

# DICE Loss

import tensorflow.keras.backend as K

'''
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # (16, 256, 256, 2)
    y_true = y_true[:, :, :, 1]
    y_pred = y_pred[:, :, :, 1]

    # (16, 256, 256)
    intersection = K.sum(y_true * y_pred, axis=(1, 2))
    dice = (2. * intersection + smooth) / (K.sum(y_true, axis=(1, 2)) + K.sum(y_pred, axis=(1, 2)) + smooth)
    return dice

'''

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    # (16, 256, 256, 2)
    y_true = y_true[:, :, :, 1]
    y_pred = y_pred[:, :, :, 1]

    # (16, 256, 256)
    tp = K.sum(K.cast(K.equal(y_true, y_pred), 'float64'))
    fn = K.sum(K.cast(K.equal(y_true, 1) & K.equal(y_pred, 0), 'float64'))
    fp = K.sum(K.cast(K.equal(y_true, 0) & K.equal(y_pred, 1), 'float64'))

    dice = (2 * tp) / (2 * tp + fn + fp + smooth)
    return dice

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def dice(y_true, y_pred):
    result = K.mean(dice_loss(y_true, y_pred))
    return result

# --------------------------------------------------------------------------------

# Tversky Loss

def tversky(y_true, y_pred, alpha=0.4, beta=0.6, smooth=1e-6):
    y_true_flat = K.flatten(y_true[:, :, :, 1])
    y_pred_flat = K.flatten(y_pred[:, :, :, 1])
    
    tp = K.sum(y_true_flat * y_pred_flat)
    fp = K.sum(y_pred_flat) - tp
    fn = K.sum(y_true_flat) - tp

    tversky_score = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    tversky_loss = 1 - tversky_score
    
    return K.mean(tversky_loss)

# --------------------------------------------------------------------------------

# Weighted Binary CrossEntropy Loss

def weighted_binary_crossentropy(y_true, y_pred, weight_positive = 0.9, weight_negative = 0.1):
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = (weight_positive * y_true * bce) + (weight_negative * (1 - y_true) * bce)
    
    return K.mean(weighted_bce)
