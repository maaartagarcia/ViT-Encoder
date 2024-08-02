import numpy as np
import keras
from keras import ops, layers
from vit_model import generate_vit_model
import pdb

class VitLayer(layers.Layer):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model

    def call(self, inputs):
        return self.vit_model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"vit_model": self.vit_model})
        return config

if __name__ == "__main__":
    inputs = np.zeros((100, 256, 256, 3))
    pdb.set_trace()
    model = generate_vit_model(inputs)
    pretrained_model = VitLayer(model)
