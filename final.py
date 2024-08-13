# Utilities
import os
import h5py
import numpy as np
import math
import time
import pdb
from utils import *
import cv2
import sys

import keras
from keras import layers, ops
import tensorflow as tf
from vit_layer import VitLayer
from vit_model import generate_vit_model

def training_callbacks():
	''' Retrieve initialized callbacks for model.fit '''
	return [
		keras.callbacks.TensorBoard(
		  log_dir=os.path.join(os.getcwd(), "unet_logs"),
		  histogram_freq=1,
		  write_images=True
		)
	]

def evaluate_model(test_file, x_test, y_test, model, pred_threshold):
    f = h5py.File(test_file, 'w')
    images = []
    masks = []
    masks_predictions = []
    predictions = model.predict(x_test)
    curr_pred = predictions[0]
    trans_curr_pred = transform_predictions(curr_pred, 0.5)

    assert x_test[0].shape == (256, 256, 3)
    assert y_test[0].shape == (256, 256, 2)
    assert curr_pred.shape == (256, 256, 2)
    assert trans_curr_pred.shape == (256, 256)

    assert np.all((curr_pred >= 0) & (curr_pred <= 1))
    assert not np.all(np.isin(curr_pred, [0, 1]))
    assert np.all((trans_curr_pred >= 0) & (trans_curr_pred <= 1))
    assert np.all(np.isin(trans_curr_pred, [0, 1]))

    for j in range(x_test.shape[0]):
        prediction = predictions[j]
        img = x_test[j] * 255.
        m = transform_predictions(y_test[j], pred_threshold)
        p = transform_predictions(prediction, pred_threshold)

        images.append(img)
        masks.append(m)
        masks_predictions.append(p)
        cv2.imwrite(f"./results/images/{j}.jpg", img)
        cv2.imwrite(f"./results/masks/{j}.png", m * 255)
        cv2.imwrite(f"./results/predictions/{j}.png", p * 255)
    
    f.create_dataset("validation_img", data = images)
    f.create_dataset("validation_labels", data = masks)
    f.create_dataset("validation_pred", data = masks_predictions)

def config():
    return dict(
        train_p = 0.8,
        val_p = 0.1,
        test_p = 0.1,
        width = 256,
        height = 256,
        channels = 3,
        kernel = (3, 3),
        pool_size = (2, 2),
        maxpool_stride = (2, 2),
        initializer = keras.initializers.HeNormal(),
        optimizer = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        batch_size = 16,
        epochs = 100,
        loss = keras.losses.BinaryCrossentropy(),
        last_kernel = (1, 1),
        pred_threshold = 0.5,
        finetuning = False,
        dataset_hdf5 = 'casia_database.hdf5',
    )

def is_config_ok(cfg):
    required_keys = ['train_p', 'val_p', 'test_p', 'width', 'height', 'channels',
                     'kernel', 'pool_size', 'maxpool_stride', 'initializer',
                     'optimizer', 'epochs', 'batch_size', 'loss', 'last_kernel', 'pred_threshold',
                     'finetuning', 'dataset_hdf5']

    for key in required_keys:
        if key not in cfg:
            print(f"Undefined {key}")
            return False

    if cfg['train_p'] + cfg['val_p'] + cfg['test_p'] != 1:
        print('Data partitioning proportions should sum 1.')
        return False

    return True

def verify_all_inputs(x_train, y_train, x_val, y_val, x_test, y_test, num_imgs):
    assert x_train.shape[1:] == (256, 256, 3)
    assert x_val.shape[1:] == (256, 256, 3)  
    assert x_test.shape[1:] == (256, 256, 3)

    assert y_train.shape[1:] == (256, 256, 2)
    assert y_val.shape[1:] == (256, 256, 2)
    assert y_test.shape[1:] == (256, 256, 2)

    assert x_train.shape[0] + x_val.shape[0] + x_test.shape[0] == num_imgs
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == num_imgs

    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    assert np.all((x_train >= 0) & (x_train <= 1))
    assert np.all((x_val >= 0) & (x_val <= 1))
    assert np.all((x_test >= 0) & (x_test <= 1))

    assert not np.all(np.isin(x_train, [0, 1]))
    assert not np.all(np.isin(x_val, [0, 1]))
    assert not np.all(np.isin(x_test, [0, 1]))

    assert np.all(np.isin(y_train, [0, 1]))
    assert np.all(np.isin(y_val, [0, 1]))
    assert np.all(np.isin(y_test, [0, 1]))

    print("xtrain", sum(x is None for x in x_train))
    print("x_test", sum(x is None for x in x_test))
    print("x_val", sum(x is None for x in x_val))

    assert np.any(x is None for x in x_train)
    assert np.any(y is None for y in y_train)
    assert np.any(x is None for x in x_val)
    assert np.any(y is None for y in y_val)
    assert np.any(x is None for x in x_test)
    assert np.any(y is None for y in y_test)

def save_test_data(x_test, y_test, file):
    if os.path.exists(file):
        print(f"File {file} already exists.")
        exit()
    
    f = h5py.File(file, 'w')
    f.create_dataset('test_img', data = x_test)
    f.create_dataset('test_labels', data = y_test)
    f.close()

@tf.keras.utils.register_keras_serializable()
class CBAMLayer(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        
        # Capas compartidas
        self.shared_dense_one = layers.Dense(channel // self.ratio, activation="relu", use_bias=False)
        self.shared_dense_two = layers.Dense(channel, use_bias=False)
        
        # Conv layer for spatial attention
        self.conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")
        
        super(CBAMLayer, self).build(input_shape)

    def call(self, x):
        # Aplicar el módulo de atención de canal
        x = self.channel_attention_module(x)
        # Aplicar el módulo de atención espacial
        x = self.spatial_attention_module(x)
        return x

    def channel_attention_module(self, x):
        # Global Average Pooling
        x1 = layers.GlobalAveragePooling2D()(x)
        x1 = self.shared_dense_one(x1)
        x1 = self.shared_dense_two(x1)
        
        # Global Max Pooling
        x2 = layers.GlobalMaxPooling2D()(x)
        x2 = self.shared_dense_one(x2)
        x2 = self.shared_dense_two(x2)
        
        # Sumar ambas características y pasar por sigmoid
        feats = x1 + x2
        feats = layers.Activation("sigmoid")(feats)
        feats = tf.reshape(feats, [-1, 1, 1, tf.shape(x)[-1]])
        feats = layers.Multiply()([x, feats])
        
        return feats

    def spatial_attention_module(self, x):
        # Average Pooling
        x1 = tf.reduce_mean(x, axis=-1, keepdims=True)
        
        # Max Pooling
        x2 = tf.reduce_max(x, axis=-1, keepdims=True)
        
        # Concatenar ambas características
        feats = layers.Concatenate()([x1, x2])
        
        # Capa Conv
        feats = self.conv(feats)
        feats = layers.Multiply()([x, feats])
        
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

# PARAMETERIZATION
# ------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
    from tensorflow.keras import backend as K

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("Error: No se ha detectado una GPU compatible. Abortando la ejecución.")
        sys.exit(1)

    cfg = config()

    if not is_config_ok(cfg):
        exit()

    f = cfg['dataset_hdf5'] 

    if not os.path.exists(f):
        print(f"Dataset not found ({f}).")
        exit()

    database = h5py.File(f, 'r')
    X = database['train_img']
    Y = database['train_labels']
    num_imgs = X.shape[0] 

    train_p = int(math.floor(num_imgs * cfg['train_p']))
    val_p = int(math.floor(num_imgs * cfg['val_p']))
    x_train, y_train = X[:train_p], Y[:train_p]
    x_val, y_val = X[train_p:(train_p + val_p)], Y[train_p:(train_p + val_p)]    
    x_test, y_test = X[(train_p + val_p):], Y[(train_p + val_p):]

    w, h, c = cfg['width'], cfg['height'], cfg['channels']
    input_shape = (num_imgs, w, h, c)
    verify_all_inputs(x_train, y_train, x_val, y_val, x_test, y_test, num_imgs)

    kernel = cfg['kernel']
    initializer = cfg['initializer']
    pool_size = cfg['pool_size'] 
    maxpool_stride = cfg['maxpool_stride']
    optimizer = cfg['optimizer']
    batch_size = cfg['batch_size'] 
    epochs = cfg['epochs']
    loss = cfg['loss']
    last_kernel = cfg['last_kernel']
    pred_threshold = cfg['pred_threshold']  
    num_classes = 2
    val_file = './hdf5/validation_results.hdf5'
    test_file = './hdf5/test_data.hdf5'
    finetuning = cfg['finetuning'] 

    # INPUT IMAGES
    # ------------------------------------------------------------------------------------------------------------------------------

    input_img = keras.Input(shape = input_shape[1:])

    # ------------------------------------------------------------------------------------------------------------------------------

    # ViT BRANCH
    # ------------------------------------------------------------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------------------------------------------------------------

    # ENCODER
    # ------------------------------------------------------------------------------------------------------------------------------

    encoder_layers = [] 

    # Layer 1: (256, 256, 3)
    x = layers.Conv2D(filters = 64, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(input_img)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = 64, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    encoder_layers.append(x)
    x = layers.MaxPool2D(pool_size = pool_size, strides = maxpool_stride, padding = 'same')(x)

    # Layer 2: (128, 128, 64)
    x = layers.Conv2D(filters = 128, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = 128, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    encoder_layers.append(x)
    x = layers.MaxPool2D(pool_size = pool_size, strides = maxpool_stride, padding = 'same')(x)

    # Layer 3: (64, 64, 128)
    x = layers.Conv2D(filters = 256, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = 256, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    encoder_layers.append(x)
    x = layers.MaxPool2D(pool_size = pool_size, strides = maxpool_stride, padding = 'same')(x)

    # Layer 4: (32, 32, 256)
    x = layers.Conv2D(filters = 512, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = 512, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    # ------------------------------------------------------------------------------------------------------------------------------

    # FUSION
    # ENCODER INPUT (32, 32, 512)
    # ------------------------------------------------------------------------------------------------------------------------------

    vit_layer = layers.Conv2DTranspose(filters = 512, strides = (2,2), kernel_size=kernel, padding = 'same')(vit_layer)
    x = layers.concatenate([x,vit_layer], axis = 3) 
    x = CBAMLayer(ratio=8)(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    # ------------------------------------------------------------------------------------------------------------------------------

    # DECODER
    # ------------------------------------------------------------------------------------------------------------------------------

    # Layer 1: (32, 32, 512)
    x = layers.Conv2DTranspose(filters = 256, 
                            kernel_size = maxpool_stride,
                            strides = pool_size,
                            padding = 'same',
                            kernel_initializer = initializer,
                            )(x)

    x = layers.concatenate([x, encoder_layers[-1]], axis = -1)

    x = layers.Conv2D(filters = 256, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = 256, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    # Layer 2: (64, 64, 256)
    x = layers.Conv2DTranspose(filters = 128, 
                            kernel_size = maxpool_stride,
                            strides = pool_size,
                            padding = 'same',
                            kernel_initializer = initializer,
                            )(x)

    x = layers.concatenate([x, encoder_layers[-2]], axis = -1)

    x = layers.Conv2D(filters = 128, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters = 128, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    # Layer 3: (128, 128, 128)
    x = layers.Conv2DTranspose(filters = 64, 
                            kernel_size = maxpool_stride,
                            strides = pool_size,
                            padding = 'same',
                            kernel_initializer = initializer,
                            )(x)

    x = layers.concatenate([x, encoder_layers[-3]], axis = -1)

    x = layers.Conv2D(filters = 64, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters = 64, kernel_size = kernel, kernel_initializer = initializer, padding = 'same')(x)
    x = layers.BatchNormalization(momentum=0.01)(x)
    x = layers.Activation('relu')(x)

    # ------------------------------------------------------------------------------------------------------------------------------

    # CLASSIFICATION
    # ------------------------------------------------------------------------------------------------------------------------------

    x = layers.Conv2D(
            filters = num_classes,
            kernel_size = last_kernel,
            kernel_initializer = initializer,
            padding = 'same',
            activation = 'sigmoid'
    )(x)

    # ------------------------------------------------------------------------------------------------------------------------------

    # EXECUTION
    # ------------------------------------------------------------------------------------------------------------------------------

    if False:
        save_test_data(x_test, y_test, test_file)

    if finetuning:
        print("Fine-tuning mode activated. Loading pre-trained model...")
        vit_enc_dec = keras.models.load_model('.tests/01_COMOFOD_200e_binary/trained_model.keras')
        initial_weights = {layer.name: layer.get_weights() for layer in vit_enc_dec.layers}
        for layer in vit_enc_dec.layers:
            print(f"Layer: {layer.name}, Trainable: {layer.trainable}")
    else:
        vit_enc_dec = keras.Model(input_img, x)
        vit_enc_dec.summary()

    vit_enc_dec.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    vit_enc_dec.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=training_callbacks()
    )

    vit_enc_dec.save(f'model/trained_model{"_finetuning" if finetuning else ""}.keras')

    import pprint

    print(f"* PARAMETERS: \n {pprint.pformat(cfg)}\n")
    evaluate_model(val_file, x_test, y_test, vit_enc_dec, pred_threshold)
    vit_enc_dec.save('./model/trained_model.keras', overwrite = True)

    # ------------------------------------------------------------------------------------------------------------------------------

    # END
    # ------------------------------------------------------------------------------------------------------------------------------

    final_weights = {layer.name: layer.get_weights() for layer in vit_enc_dec.layers}

    for name in initial_weights:
        if not np.array_equal(initial_weights[name], final_weights[name]):
            print(f"Weights for layer {name} have been updated.")
        else:
            print(f"Weights for layer {name} have not changed.")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total employed time: {round(total_time / 60, 2)} (min)")

    # ------------------------------------------------------------------------------------------------------------------------------