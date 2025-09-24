
# examples/tf_keras_example.py
import os, numpy as np, tensorflow as tf
from PIL import Image
from mobile_gradio_classifier import MobileClassifierApp

MODEL_PATH = os.environ.get("MODEL_PATH", "model.keras")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
classes = [line.strip() for line in open("classes.txt") if line.strip()]
model = tf.keras.models.load_model(MODEL_PATH)

def _preprocess_rgb(pil: Image.Image):
    arr = tf.image.resize(np.array(pil.convert("RGB")), (IMG_SIZE, IMG_SIZE)).numpy()
    arr = arr.astype("float32")/255.0
    return arr[None, ...]

def predict_fn(pil: Image.Image):
    y = model.predict(_preprocess_rgb(pil), verbose=0)[0]
    if (y < 0).any() or not np.allclose(y.sum(), 1.0, atol=1e-3):
        y = tf.nn.softmax(y).numpy()
    return {c: float(p) for c,p in zip(classes, y)}

app = MobileClassifierApp(classes, predict_fn=predict_fn)
app.launch(share=True)
