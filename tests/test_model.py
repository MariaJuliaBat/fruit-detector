import os
import tensorflow as tf

def test_model_loading():
    model_path = os.path.join("model", "fruit_model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    assert model is not None

def test_labels_loading():
    labels_path = os.path.join("model", "labels.txt")
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    assert isinstance(labels, list)
    assert len(labels) > 0
