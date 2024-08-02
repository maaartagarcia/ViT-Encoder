import keras
from keras import layers, ops
import matplotlib.pyplot as plt
import pdb
import cv2
import tensorflow as tf
import numpy as np
import math
import h5py
import os
from utils import transform_masks, transform_predictions, validate_input_images, dice, tversky
from utils import weighted_binary_crossentropy
from keras.callbacks import ModelCheckpoint
from vit_layer import VitLayer
from vit_model import generate_vit_model

from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Input
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Multiply

import sys
import time

start_time = time.time()

os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
from tensorflow.keras import backend as K

# comprobar existencia de GPUs disponibles
physical_devices = tf.config.list_physical_devices('GPU')

# listar todas las GPUs disponibles
print(tf.config.list_physical_devices('GPU'))

# configurar tensorflow para que vaya utilizando memoria de la GPU segun
# la necesite en la GPU indicada, si no, reserva toda la memoria de
# la GPU
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
else:
    print("Error: No se ha detectado una GPU compatible. Abortando la ejecución.")
    sys.exit(1)

@tf.keras.utils.register_keras_serializable()
class CBAMLayer(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        
        # Capas compartidas
        self.shared_dense_one = Dense(channel // self.ratio, activation="relu", use_bias=False)
        self.shared_dense_two = Dense(channel, use_bias=False)
        
        # Conv layer for spatial attention
        self.conv = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")
        
        super(CBAMLayer, self).build(input_shape)

    def call(self, x):
        # Aplicar el módulo de atención de canal
        x = self.channel_attention_module(x)
        # Aplicar el módulo de atención espacial
        x = self.spatial_attention_module(x)
        return x

    def channel_attention_module(self, x):
        # Global Average Pooling
        x1 = GlobalAveragePooling2D()(x)
        x1 = self.shared_dense_one(x1)
        x1 = self.shared_dense_two(x1)
        
        # Global Max Pooling
        x2 = GlobalMaxPooling2D()(x)
        x2 = self.shared_dense_one(x2)
        x2 = self.shared_dense_two(x2)
        
        # Sumar ambas características y pasar por sigmoid
        feats = x1 + x2
        feats = Activation("sigmoid")(feats)
        feats = tf.reshape(feats, [-1, 1, 1, tf.shape(x)[-1]])
        feats = Multiply()([x, feats])
        
        return feats

    def spatial_attention_module(self, x):
        # Average Pooling
        x1 = tf.reduce_mean(x, axis=-1, keepdims=True)
        
        # Max Pooling
        x2 = tf.reduce_max(x, axis=-1, keepdims=True)
        
        # Concatenar ambas características
        feats = Concatenate()([x1, x2])
        
        # Capa Conv
        feats = self.conv(feats)
        feats = Multiply()([x, feats])
        
        return feats

    def get_config(self):
        config = super(CBAMLayer, self).get_config()
        config.update({'ratio': self.ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

@keras.saving.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        images = K.cast(images, 'float32')

        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(patch_size=config['patch_size'])

@keras.saving.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(num_patches=config['num_patches'], projection_dim=config['projection_dim'])

def resUnit(input_layer, i, nbF):
    # Input Layer, number of layer, number of filters to be applied
    x = layers.BatchNormalization(momentum=0.01)(input_layer)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = nbF, kernel_size = kernel, activation = None, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters = nbF, kernel_size = kernel, activation = None, padding = 'same')(x)

    return layers.add([input_layer, x])

testing = False
nbFilter = 32 # Filter size
kernel = (3, 3)
pool_kernel = (2, 2) # Max Pooling Kernel Size
# Down sample Kernel to convert Encoder and ViT's output to (16, 16, 32) before Decoder
# Down sample Kernel to convert Decoder to two class maps
down_kernel = (1,1) 
batch_size = 16
outSize = 16
upsampling_factor = (4, 4)
num_classes = 2
epochs = 600
num_imgs = None # Assigned when reading the data file 
input_shape = None
pt = 0.8 # Proportion of images used for training
zt = 0.1
tt = 0.1
# Input shape: width, height, channels
w = 256
h = 256
c = 3

if testing:
    if len(sys.argv) != 2:
        print('For testing, a hdf5 file with test_img, and test_labels datasets must be passed as an argument.')
        exit()

    arg_test_file = sys.argv[1]

    if not os.path.exists(arg_test_file):
        raise ValueError("Passed test file ({arg_test_file}) should exist.")

# DATA PREPARATION
# ------------------------------------------------------------------------------------------------------------------------------

# from keras.datasets import mnist
# from keras.datasets import cifar10

# (x_train, _), (x_test, _) = mnist.load_data()
# (x_train, _), (x_test, _) = cifar10.load_data()

# imgs_file = '/scratch.local2/al404273/m08/data/comofod_and_casia_finetuning.hdf5'
imgs_file = '/scratch.local2/al404273/m08/data/casia_final_data.hdf5'
# imgs_file = '/scratch.local2/al404273/m08/data/COMOFOD_FINAL_DATA_VIT.hdf5'
# imgs_file = '/scratch.local2/al404273/m08/data/mfc18_dresden_data.hdf5'
# imgs_file = './hdf5/training_01.hdf5'
images_label = "train_img"
masks_label = "train_labels"

valid = validate_input_images(imgs_file, images_label, masks_label)

if not valid:
    print('Exiting...')
    exit()

f = h5py.File(imgs_file, 'r')

X = f[images_label]
Y = f[masks_label]  

num_imgs = X.shape[0]
num_masks = Y.shape[0]  

if num_imgs != num_masks:
    print("Fail. Not the same amount of images an masks in hdf5 file.")
    exit()

train_p = int(math.floor(num_imgs * pt)) #  Train proportion 
test_p = int(math.floor(num_imgs * (tt))) #  Validation proportion 
final_p = int(math.floor(num_imgs * (zt)))

print("Total images: ", num_imgs)
print("Training images: ", train_p)
print("Validation images: ", test_p)
print("Test images: ", final_p)

if train_p + test_p + final_p > num_imgs:
    print("Fail. Invalid training and validation proportions. Those values should match.")
    exit()

x_train, y_train = X[:train_p], Y[:train_p]
x_test, y_test = X[train_p:(train_p + test_p)], Y[train_p:(train_p + test_p)]    
zx_test, zy_test = X[(train_p + test_p):], Y[(train_p + test_p):]  

xo_train, yo_train = x_train, y_train
xo_test, yo_test  = x_test, y_test 
zxo_test, zyo_test = zx_test, zy_test

y_train_conv = np.zeros((train_p, 256, 256, 2))
y_test_conv = np.zeros((test_p, 256, 256, 2))
zy_test_conv = np.zeros((final_p, 256, 256, 2))

for i in range(train_p):
    y_train_conv[i] = transform_masks(y_train[i]) 
    
for i in range(test_p):
    y_test_conv[i] = transform_masks(y_test[i])

for i in range(final_p):
    zy_test_conv[i] = transform_masks(zy_test[i]) 

y_train = y_train_conv
y_test = y_test_conv
zy_test = zy_test_conv

# Normalized images only
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.
zx_test = zx_test.astype('float32') / 255.
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
zy_test = zy_test.astype('float32')

print("\n***** AUTO ENCODER  *****")
print("Shape Train Img: ", x_train.shape, " with data type: ", type(x_train[0][0][0][0]))
print("Shape Train masks: ", y_train.shape, " with data type: ", type(y_train[0][0][0][0]))
print("Shape Test Imgs: ", x_test.shape, " with data type: ", type(x_test[0][0][0][0]))
print("Shape Test Masks: ", y_test.shape, " with data type: ", type(y_test[0][0][0][0]))

input_shape = (num_imgs, w, h, c)

print("--> Model's input shape: ", input_shape)

# ------------------------------------------------------------------------------------------------------------------------------

# input_img = keras.Input(shape=(28, 28, 1))
input_img = keras.Input(shape = input_shape[1:] )

print("***** VIT BRANCH *****")
print("Shape of ViT Branch generated features")

weight_decay_vit = 0.0001
image_size_vit = 256
patch_size_vit = 32
num_patches = (image_size_vit  // patch_size_vit) ** 2
projection_dim_vit = 64
num_heads_vit = 4
transformer_units_vit = [
    projection_dim_vit * 2,
    projection_dim_vit,
]

transformer_layers_vit = 8
mlp_head_units_vit = [
    2048,
    1024,
]

patches_vit = Patches(patch_size_vit)(input_img)
encoded_patches_vit = PatchEncoder(num_patches, projection_dim_vit)(patches_vit)

for _ in range(transformer_layers_vit):
    x1_vit = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_vit)
    attention_output_vit = layers.MultiHeadAttention(
    num_heads=num_heads_vit, key_dim=projection_dim_vit, dropout=0.1
        )(x1_vit, x1_vit)
    x2_vit = layers.Add()([attention_output_vit, encoded_patches_vit])
    x3_vit = layers.LayerNormalization(epsilon=1e-6)(x2_vit)
    x3_vit = mlp(x3_vit, hidden_units=transformer_units_vit, dropout_rate=0.1)
    encoded_patches_vit = layers.Add()([x3_vit, x2_vit])

representation_vit = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_vit)
representation_vit = layers.Flatten()(representation_vit)
features_vit = representation_vit
vit_layer = layers.Reshape((16, 16, 16))(features_vit)

# ENCODER
# ------------------------------------------------------------------------------------------------------------------------------

encoder_layers = []

# layer 1 -> Input (256, 256, 3) // Output (128, 128, 32)
# (Before) layer1 = slim.conv2d( input_layer, nbFilter,[3,3], normalizer_fn = slim.batch_norm, scope = 'conv_' + str( 0 )  )
# (Before) layer1 = slim.conv2d( input_layer, nbFilter,[3,3], normalizer_fn = slim.batch_norm, scope = 'conv_' + str( 0 )  )
# (Original) x = layers.Conv2D(16, kernel, activation='relu', padding='same')(input_img)
x = layers.Conv2D(filters = nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(input_img)
x = layers.BatchNormalization(momentum=0.01)(x)
x = resUnit(x, 1, nbFilter)
x = layers.ReLU()(x)
encoder_layers.append(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# layer 2 -> Input (128, 128, 32) // Output (64, 64, 64)
x = layers.Conv2D(filters = 2 * nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization(momentum=0.01)(x)
x = resUnit(x, 2, 2 * nbFilter)
x = layers.ReLU()(x)
encoder_layers.append(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# (Original) x = layers.Conv2D(4*nbFilter, kernel, activation='relu', padding='same')(x)
# (Original) encoded = layers.MaxPooling2D(pool_kernel, padding='same')(x)

# layer 3 -> Input (64, 64, 64) // Output (32, 32, 128)
x = layers.Conv2D(filters = 4 * nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization(momentum=0.01)(x)
x = resUnit(x, 3, 4 * nbFilter)
x = layers.ReLU()(x)
encoder_layers.append(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# layer 4 -> Input (32, 32, 128) // Output(16, 16, 256)
x = layers.Conv2D(filters = 8 * nbFilter, kernel_size = kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization(momentum=0.01)(x)
x = resUnit(x, 4, 8 * nbFilter)
x = layers.ReLU()(x)
encoder_layers.append(x)
x = layers.MaxPooling2D(pool_size = pool_kernel, padding='same')(x)

# Final representation (16, 16, 256)
# ------------------------------------------------------------------------------------------------------------------------------

# CONCATENATE ENCODER + ViT --> Input (16, 16, 32) + (16, 16, 32) // Output (16, 16, 64)
# ------------------------------------------------------------------------------------------------------------------------------

# ENCODER
# Layer from Encoder goes from (16, 16, 256) to (16, 16, 32)
# ViT layer's output should also have (16, 16, 32) shape

x = layers.Conv2D(filters = nbFilter, kernel_size = down_kernel, activation = None,  padding = 'same')(x)
x = layers.BatchNormalization(momentum=0.01)(x)
x = layers.ReLU()(x)

# (Before) top = slim.conv2d(layer4,nbFilter,[1,1], normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_top')
# (Before) top = tf.nn.relu(top)
# (Before)concatenate both lstm features and image features
# (Before) joint_out=tf.concat([top,lstm_out],3)

# ViT
# ...

# CONCATENATE
x = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)

x = layers.concatenate([x,vit_layer], axis = 3)

# ------------------------------------------------------------------------------------------------------------------------------

x = CBAMLayer(ratio=8)(x)

# DECODER --> Input (16, 16, 32) // Output (256, 256, 2)
# ------------------------------------------------------------------------------------------------------------------------------
'''
x = layers.Conv2DTranspose(filters=nbFilter,  # Number of filters (channels) you want in the output
                    kernel_size=upsampling_factor,
                    strides=upsampling_factor,
                    padding='same',  # Padding to ensure output size is as expected
                    activation='relu')(x)
x = layers.concatenate([x, encoder_layers[-1]] , axis = -1)
# (16, 64, 64, 2)
# Conv2D kernel size modified from (1,1) to (3,3)
x = layers.Conv2D(filters = num_classes, kernel_size = kernel, activation = None, padding='same')(x)
x = layers.BatchNormalization(momentum=0.01)(x)
x = layers.ReLU()(x)

# Upsampling to batch size (16, 256, 256, 2)
x = layers.UpSampling2D(size = upsampling_factor, interpolation = 'bilinear')(x)
# Added
x = layers.Conv2D(filters = num_classes, kernel_size = kernel, activation = 'softmax', padding='same')(x)
# x = layers.BatchNormalization()(x)
# x = layers.ReLU()(x)
'''

# ------------------------------------------------------------------------------------------------------------------------------

x = layers.Conv2DTranspose(filters=nbFilter,
                    kernel_size=kernel,
                    strides=pool_kernel,
                    padding='same',
                    activation='relu')(x)
print(f"Should be (32, 32, 32) and its {x}")
x = layers.Conv2D(8 * nbFilter, (1, 1), padding='same', activation='relu')(x)
print(f"Should be (32, 32, 256) and its {x}")
x = layers.concatenate([x, encoder_layers[-1]], axis = -1)
print(f"Should be (32, 32, 512) and its {x}")
x = Conv2D(filters=4 * nbFilter, kernel_size=kernel, activation='relu', padding='same')(x)
print(f"Should be (32, 32, 128) and its {x}")

x = layers.Conv2DTranspose(filters=2*nbFilter,
                    kernel_size=kernel,
                    strides=pool_kernel,
                    padding='same',
                    activation='relu')(x)
x = layers.Conv2D(4 * nbFilter, (1, 1), padding='same', activation='relu')(x)
print(f"Should be (64, 64, 128) and its {x}")
x = layers.concatenate([x, encoder_layers[-2]], axis = -1)
print(f"Should be (64, 64, 256) and its {x}")
x = Conv2D(filters=2*nbFilter, kernel_size=kernel, activation='relu', padding='same')(x)
print(f"Should be (64, 64, 64) and its {x}")

x = layers.Conv2DTranspose(filters=4*nbFilter,
                    kernel_size=kernel,
                    strides=pool_kernel,
                    padding='same',
                    activation='relu')(x)
x = layers.Conv2D(2 * nbFilter, (1, 1), padding='same', activation='relu')(x)
print(f"Should be (128, 128, 64) and its {x}")
x = layers.concatenate([x, encoder_layers[-3]], axis = -1)
print(f"Should be (128, 128, 128) and its {x}")
x = Conv2D(filters=nbFilter, kernel_size=kernel, activation='relu', padding='same')(x)
print(f"Should be (128, 128, 32) and its {x}")

x = layers.Conv2DTranspose(filters=8 * nbFilter,
                    kernel_size=kernel,
                    strides=pool_kernel,
                    padding='same',
                    activation='relu')(x)
x = layers.Conv2D(nbFilter, (1, 1), padding='same', activation='relu')(x)
print(f"Should be (256, 256, 32) and its {x}")
x = layers.concatenate([x, encoder_layers[-4]], axis = -1)
print(f"Should be (256, 256, 64) and its {x}")

x = layers.Conv2D(filters = num_classes, kernel_size = kernel, activation = 'softmax', padding='same')(x)

# SET THE MODEL CONFIGURATIONS
# ------------------------------------------------------------------------------------------------------------------------------

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, x)

# This is the size of our encoded representations
# encoding_dim = 128  # latent representation is (4, 4, 8) i.e. 128-dimensional

# Show the summary of the model architecture
autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'] )
# autoencoder.compile(optimizer=keras.optimizers.SGD(), loss=dice, metrics = ['accuracy'] )
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')
# autoencoder.compile(optimizer='adam', loss=dice, metrics = ['accuracy'] )
autoencoder.compile(optimizer='adam', loss=tversky, metrics = ['accuracy'] )
# autoencoder.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics = ['accuracy'] )

# ------------------------------------------------------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------------------------------------------------------

if not testing:
    '''
    model_loaded = '../27_CMFD_600e_binary_FAIL/trained_model.keras'
    if os.path.exists(model_loaded):
        autoencoder = keras.saving.load_model(model_loaded)

        print(f"Finetuning model...")
    else:
        print("No finetuning file found.")

    checkpoint = ModelCheckpoint(
        'trained_model.keras', 
        monitor='val_loss', 
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )
    '''
    history = autoencoder.fit(
                            x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(x_test, y_test),
                            )
     

# ------------------------------------------------------------------------------------------------------------------------------

# LEARNING CURVES
# ------------------------------------------------------------------------------------------------------------------------------

if testing:
    autoencoder.load_weights("./model/trained_model.keras")
    # autoencoder.keras.load_model("./model/trained_model.keras")
    # autoencoder.load_weights("../01_CMFD_10000i_100e/trained_model.keras")
    # autoencoder = keras.models.load_model("./model/trained_model.keras")
    # autoencoder = keras.models.load_model("./COVERAGE_metrics/trained_model.keras")

if not testing:
    '''
    plt.figure(2)
    plt.plot(range(1,epochs+1), history.history['loss'])
    plt.plot(range(1,epochs+1), history.history['val_loss'])
    #plt.xticks(range(1,epochs+1))
    plt.xlim(1,epochs)
    plt.ylim(0, 0.25)
    plt.show()
    '''

    # Encode and decode some digits
    # Note that we take them from the *test* set

    decoded_imgs = autoencoder.predict(zx_test)

    n = decoded_imgs.shape[0]   # How many digits we will display

    while n > decoded_imgs.shape[0]:
        print("Fail. Testing images cannot be greater than x_test.")
        response = input('New number of testing images? (number/no) ')
        if n == 'no':
            exit()
        n = int(response)

    result_file = './hdf5/validation_results.hdf5'
    print("Important. Testing of ", n, " images (Original images, masks, and model predictions) will be saved in ", result_file, " file.")

    if os.path.exists(result_file):
        print("Fail. ", result_file, " already exists. ")
        response = input("Want to delete? (s/n): ")
        if response == "s":
            os.remove(result_file)

    f = h5py.File(result_file, 'w')

    # plt.figure(figsize=(20, 4))

    images = []
    predictions = []
    masks = []

    for i in range(n):
        image = zx_test[i] * 255
        mask = zyo_test[i] * 255  

        # Added
        pred = transform_predictions(decoded_imgs[i], 0.5) * 255
        cv2.imwrite('./results/images/original_' + str(i) + '.jpg', image) 

        images.append(image)
        masks.append(mask)
        predictions.append(pred)

        cv2.imwrite('./results/masks/mask_' + str(i) + '.png', mask) 
        cv2.imwrite('./results/predictions/predicted_' + str(i) + '.png', pred) 

        '''
        # Display original
        ax = plt.subplot(2, n, i + 1)
        # plt.imshow(x_test[i].reshape(28, 28))
        plt.imshow(zx_test[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.imshow(pred, cmap = 'gray')
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        '''

    f.create_dataset("validation_img", shape = (n, w, h, c), data = images)
    f.create_dataset("validation_labels", shape = (n, w, h), data = masks)
    f.create_dataset("validation_pred", shape = (n, w, h), data = predictions)

    autoencoder.save('./model/trained_model.keras', overwrite = True)

if testing:
    '''
    ni = zx_test.shape[0]  
    accuracy = autoencoder.evaluate(x = zx_test, y = zy_test)
    preditions = autoencoder.predict(zx_test)
    print("Accuracy: ", accuracy[1]) 
    '''

    test_file = h5py.File(arg_test_file, 'r')
    zx_test = np.array(test_file["test_img"])
    zyo_test = np.array(test_file["test_labels"])

    num_test_images = zx_test.shape[0]
    num_test_masks = zyo_test.shape[0]

    zx_test = zx_test.astype('float32')
    zyo_test = zyo_test.astype('float32')

    if num_test_images != num_test_masks or num_test_images == 0:
        print('Number of images ({num_test_images}) and masks ({num_test_masks}) must be equal and more than zero.')
        exit()

    decoded_imgs = autoencoder.predict(zx_test)
    num_predictions = decoded_imgs.shape
    n = num_predictions[0]   # How many digits we will display

    print("Predictions generated, shape: {num_predictions}")

    while n > num_predictions[0]:
        print("Fail. Testing images cannot be greater than x_test.")
        response = input('New number of testing images? (number/no) ')

        if n == 'no':
            exit()

        n = int(response)

    result_file = './hdf5/test_results.hdf5'

    if os.path.exists(result_file):
        print("Fail. ", result_file, " already exists. Delete and try again.")
        exit()

    # plt.figure(figsize=(20, 4))

    images = []
    predictions = []
    masks = []

    for i in range(n):
        image = zx_test[i] 
        mask = zyo_test[i]

        if image.shape != (256, 256, 3):
            raise ValueError(f"Images should have (256, 256, 3) shape, not {image.shape}")

        if mask.shape != (256, 256):
            raise ValueError(f"Masks should have (256, 256) shape, not {mask.shape}")

        if np.any((mask != 0) & (mask != 1)):
            print('Warning: Masks should be binary masks.')

            for z in range(256):
                for t in range(256):
                    if mask[z][t] == 255:
                        mask[z][t] = 1

        if np.any((mask != 0) & (mask != 1)):
            raise ValueError('All masks should be binary masks.')

        # Important. Threshold is 0.5 for classification
        if decoded_imgs[i].shape != (256, 256, 2):
            raise ValueError(f"Ouput predictions from the model should have (256, 256, 2) shape, not {decoded_imgs[i].shape}.")

        pred = transform_predictions(decoded_imgs[i], 0.5)
        
        if pred.shape != (256, 256):
            raise ValueError(f"Ouput predictions after transformation should have (256, 256) shape, not {pred.shape}.")

        pred = pred.astype('float32')
        cv2.imwrite('./testing/images/original_' + str(i) + '.tif', image) 
        cv2.imwrite('./testing/masks/mask_' + str(i) + '.tif', mask * 255)
        cv2.imwrite('./testing/predictions/predicted_' + str(i) + '.tif', pred * 255) 

        images.append(image)
        masks.append(mask)
        predictions.append(pred)

        '''
        # Display original
        ax = plt.subplot(2, n, i + 1)
        # plt.imshow(x_test[i].reshape(28, 28))
        plt.imshow(zx_test[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.imshow(pred, cmap = 'gray')
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        '''

    result_file = 'testing_results.hdf5'
    print("Important. Testing of ", n, " images (Original images, masks, and model predictions) will be saved in ", result_file, " file.")

    f = h5py.File(result_file, 'w')
    f.create_dataset("validation_img", shape = (n, w, h, c), data = images)
    f.create_dataset("validation_labels", shape = (n, w, h), data = masks)
    f.create_dataset("validation_pred", shape = (n, w, h), data = predictions)

end_time = time.time()

total_time_employed = end_time - start_time

print(f"Total employed time: {total_time_employed}")

# ------------------------------------------------------------------------------------------------------------------------------
