from common_configs import TrainerA2C, TrainerConfig, ClassicConfConfig
from scripts.train import prepare_game
import tensorflow as tf
import keras

from keras.models import load_model

h5_path = "/home/mwm/repositories/lartpc/lartpc2D-rl/mlruns/3/01bebe89610e4bb98f6615fb19e26945/artifacts/target_models/data/model.h5"
model = load_model(h5_path, custom_objects={'tf':tf})
tf.keras.utils.plot_model(
    model,
    to_file="assets/pics/model.png",
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)
