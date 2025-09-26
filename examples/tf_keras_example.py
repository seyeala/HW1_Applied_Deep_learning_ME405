
# examples/tf_keras_example.py
import os, numpy as np, tensorflow as tf
from pathlib import Path
from PIL import Image
from mobile_gradio_classifier import MobileClassifierApp

MODEL_PATH = os.environ.get("MODEL_PATH", "model.keras")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
DEFAULT_CLASSES = ["class_one", "class_two"]
classes_path = Path(__file__).with_name("classes.txt")
if classes_path.exists():
    classes = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()] or DEFAULT_CLASSES
else:
    classes = DEFAULT_CLASSES
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
